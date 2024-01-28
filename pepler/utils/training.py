import math
from logging import Logger

import torch
import torch.nn as nn
from tqdm import tqdm

import wandb

from .metrics import class_metrics


def exp(x):
    return math.exp(x) if x < 1e2 else float("inf")


###############################################################################
# Training, evaluate, generate functions
###############################################################################
def safe_multiloss(
    text_loss: float,
    overall_rating_loss: float,
    aspect_rating_loss: float,
    text_reg: float,
    overall_rating_reg: float,
    aspect_rating_reg: float,):
    # text_reg * text_loss + aspect_rating_reg * aspect_rating_loss + overall_rating_reg * overall_rating_loss
    # if any of the loss is nan or inf, set it to 0
    if math.isnan(text_loss) or math.isinf(text_loss): text_loss = 0.0
    if math.isnan(overall_rating_loss) or math.isinf(overall_rating_loss): overall_rating_loss = 0.0
    if math.isnan(aspect_rating_loss) or math.isinf(aspect_rating_loss): aspect_rating_loss = 0.0

    return text_reg * text_loss + aspect_rating_reg * aspect_rating_loss + overall_rating_reg * overall_rating_loss

def train(
    data_loader,
    model,
    logger: Logger,
    device: str,
    optimizer: torch.optim,
    gradient_accumulation_steps: int,
    rating_criterion: torch.nn.functional,
    do_rating: str,
    text_reg: float,
    overall_rating_reg: float,
    aspect_rating_reg: float,
    log_interval: int = 200,
):
    # Turn on training mode which enables dropout.
    model.train()
    text_loss = 0.0
    overall_rating_loss = 0.0
    aspect_rating_loss = 0.0
    total_sample = 0
    for batch_idx, batch in enumerate(tqdm(data_loader), start=1):
        user = batch.user.to(device)
        item = batch.item.to(device)
        aspect = batch.aspect.to(device)
        rating = batch.rating.to(device).squeeze(-1)
        seq = batch.seq.to(device)  # (batch_size, seq_len)
        mask = batch.mask.to(device)
        overall_rating = batch.overall_rating.to(device)
        aspect_score = batch.aspect_score.to(device)
        seq = batch.seq.to(device)  # (batch_size, seq_len)
        mask = batch.mask.to(device)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()
        outputs = model(user, item, aspect, seq, mask, aspect_score)
        # NLL loss
        t_loss = loss = outputs.loss
        if do_rating.lower() != "none":
            predicted_aspect_rating = outputs.aspect_rating
            predicted_overall_rating = outputs.overall_rating
            ra_loss = rating_criterion(predicted_aspect_rating, rating)
            ro_loss = rating_criterion(predicted_overall_rating, overall_rating)
            loss = safe_multiloss(
                text_loss=t_loss,
                overall_rating_loss=ro_loss,
                aspect_rating_loss=ra_loss,
                text_reg=text_reg,
                overall_rating_reg=overall_rating_reg,
                aspect_rating_reg=aspect_rating_reg,
            )
            loss = loss / gradient_accumulation_steps
        loss.backward()
        if batch_idx % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        batch_size = user.size(0)
        text_loss += batch_size * loss.item()
        overall_rating_loss += (
            batch_size * ro_loss.item() if do_rating.lower() != "none" else 0.0
        )
        aspect_rating_loss += (
            batch_size * ra_loss.item() if do_rating.lower() != "none" else 0.0
        )
        total_sample += batch_size

        if batch_idx % log_interval == 0 or batch == len(data_loader) - 1:
            cur_t_loss = text_loss / total_sample
            cur_ro_loss = (
                overall_rating_loss / total_sample
                if do_rating.lower() != "none"
                else 0.0
            )
            cur_ra_loss = (
                aspect_rating_loss / total_sample
                if do_rating.lower() != "none"
                else 0.0
            )
            logger.info(
                f"text loss: {cur_t_loss} | overall rating loss: {cur_ro_loss} | aspect rating loss: {cur_ra_loss} |\
                        {batch_idx}/{len(data_loader)} batches"
            )
            logger.info(
                "text ppl {:4.4f} | {:5d}/{:5d} batches".format(
                    exp(cur_t_loss), batch_idx, len(data_loader)
                )
            )
            wandb.log(
                {
                    "step": batch_idx,
                    "train_t_loss": cur_t_loss,
                    "train_ro_loss": cur_ro_loss,
                    "train_ra_loss": cur_ra_loss,
                    "train_t_ppl": exp(cur_t_loss),
                }
            )

            text_loss = 0.0
            total_sample = 0


def evaluate(
    data_loader,
    model,
    logger: Logger,
    device: str,
    rating_criterion: torch.nn.functional,
    do_rating: str,
    class_num:int=3,
    average:str='macro'
):
    # Turn on evaluation mode which disables dropout.
    model.to(device)
    model.eval()
    text_loss = 0.0
    overall_rating_loss = 0.0
    aspect_rating_loss = 0.0
    total_sample = 0

    with torch.no_grad():
        global_predicted_aspect_rating = []
        global_predicted_overall_rating = []
        for batch_idx, batch in enumerate(tqdm(data_loader), start=1):
            user = batch.user.to(device)  # (batch_size,)
            item = batch.item.to(device)  # (batch_size,)
            aspect = batch.aspect.to(device)  # (batch_size,)
            rating = batch.rating.to(device)  # (batch_size,)
            aspect_score = batch.aspect_score.to(device)  # (batch_size, aspect_num)
            overall_rating = batch.overall_rating.to(device)
            seq = batch.seq.to(device)  # (batch_size, seq_len)
            mask = batch.mask.to(device)
            outputs = model(user, item, aspect, seq, mask, aspect_score)
            t_loss = outputs.loss


            if do_rating.lower() != "none":
                predicted_aspect_rating = outputs.aspect_rating
                predicted_overall_rating = outputs.overall_rating
                global_predicted_aspect_rating.extend(predicted_aspect_rating.tolist())
                global_predicted_overall_rating.extend(predicted_overall_rating.tolist())
                ra_loss = rating_criterion(predicted_aspect_rating, rating)
                ro_loss = rating_criterion(predicted_overall_rating, overall_rating)

            batch_size = user.size(0)
            text_loss += batch_size * t_loss.item()
            overall_rating_loss += (
                batch_size * ra_loss.item() if do_rating.lower() != "none" else 0.0
            )
            aspect_rating_loss += (
                batch_size * ro_loss.item() if do_rating.lower() != "none" else 0.0
            )
            total_sample += batch_size
    f1, prec, recall, acc = class_metrics(
        [(r,p) for r,p in zip(global_predicted_overall_rating, global_predicted_aspect_rating)],
        class_num=class_num, average=average)
    logger.info(f"aspect rating f1: {f1}, prec: {prec}, recall: {recall}, acc: {acc}")


    return (
        text_loss / total_sample,
        overall_rating_loss / total_sample,
        aspect_rating_loss / total_sample,
    )


def generate(data_loader, model, logger, device, temperature=1.0) -> tuple:
    """Do aspect text generation and aspect rating prediction
    TODO: this is not a complete version in testing stage, maybe
    write a way to generate aspect text and aspect ratings, and then
    use the predicted aspect ratings for overall rating prediction

    Parameters
    ----------
    data : a Batchify object
    model : huggingface model
    logger : Logger
    device : str
    temperature : float, optional

    Returns
    -------
    tuple of words indices and ratings
    """

    if not (1e-3 <= temperature <= 1):
        logger.info("temperature should be 1e-3 <=temp <= 1; manually set to 1e-3")

    model.eval()  # Turn on evaluation mode which disables dropout.
    idss_predict = []
    aspect_rating_predict = []

    with torch.no_grad():
        for _, batch in enumerate(tqdm(data_loader), start=1):
            user = batch.user.to(device)  # (batch_size,)
            item = batch.item.to(device)
            aspect = batch.aspect.to(device)
            aspect_score = batch.aspect_score.to(device)
            text = batch.seq[:, :1].to(device)  # bos, (batch_size, 1)
            for idx in range(batch.seq.size(1)):
                # produce a word at each step

                outputs = model(
                    user, item, aspect, text, mask=None, aspect_score=aspect_score
                )
                if idx == 0:
                    aspect_rating_predict.extend(outputs.aspect_rating.tolist())
                # last_token = outputs.logits[:, -1, :]  # the last token, (batch_size, ntoken)
                # word_prob = torch.softmax(last_token, dim=-1)
                # token = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1), pick the one with the largest probability

                last_token = outputs.logits[:, -1, :] / (temperature + 1e-10)
                # the last token, (batch_size, ntoken)
                word_prob = torch.softmax(last_token, dim=-1)  # (batch_size, ntoken)
                token = torch.multinomial(
                    word_prob, 1
                )  # Sample from the multinomial distribution

                text = torch.cat([text, token], 1)  # (batch_size, len++)
            ids = text[:, 1:].tolist()  # remove bos, (batch_size, seq_len)
            idss_predict.extend(ids)
    return idss_predict, aspect_rating_predict


def train_loop(
    model,
    train_loader,
    val_loader,
    logger,
    device,
    endure_times,
    model_path,
    freeze_plm_only=False,
    epochs: int = 50,
    log_interval: int = 200,
    optimizer: torch.optim = None,
    gradient_accumulation_steps: int = 1,
    rating_criterion: torch.nn.functional = None,
    do_rating: str = None,
    text_reg: float = None,
    overall_rating_reg: float = None,
    aspect_rating_reg: float = None,
):
    if epochs == 0:
        logger.info("Skip training | epochs == 0")
        return model

    if freeze_plm_only:
        title = "Tuning Prompt Only"
    else:
        title = "Tuning Prompt, PLM and Rating"
    logger.info(title)
    # Loop over epochs.
    if freeze_plm_only:
        model.freeze_plm_only()
    else:
        model.unfreeze()

    best_val_loss = float("inf")
    endure_count = 0
    for epoch in range(1, epochs + 1):
        logger.info("epoch {}".format(epoch))
        train(
            data_loader=train_loader,
            model=model,
            logger=logger,
            device=device,
            optimizer=optimizer,
            gradient_accumulation_steps=gradient_accumulation_steps,
            rating_criterion=rating_criterion,
            do_rating=do_rating,
            text_reg=text_reg,
            overall_rating_reg=overall_rating_reg,
            aspect_rating_reg=aspect_rating_reg,
            log_interval=log_interval,
        )

        val_t_loss, val_ro_loss, val_ra_loss = evaluate(
            val_loader, model, logger, device, rating_criterion, do_rating
        )
        val_loss = val_t_loss + val_ra_loss + val_ro_loss
        wandb.log(
            {
                "epoch": epoch,
                "val_t_loss": val_t_loss,
                "val_ro_loss": val_ro_loss,
                "val_ra_loss": val_ra_loss,
                "val_t_ppl": exp(val_t_loss),
                "val_loss": val_loss,
            }
        )

        logger.info(
            "text ppl {:4.4f} | valid t loss {:4.4f} | valid ro loss {:4.4f} | valid ra loss {:4.4f} on validation.".format(
                exp(val_t_loss), val_t_loss, val_ro_loss, val_ra_loss
            )
        )

        # Save the model if the validation loss is the best we've seen so far.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            with open(model_path, "wb") as f:
                torch.save(model, f)
        else:
            endure_count += 1
            logger.info("Endured {} time(s)".format(endure_count))
            if endure_count == endure_times:
                print("Cannot endure it anymore | Exiting from early stop")
                break
    logger.info(f"Tuning takes {epoch} epochs, reaching val loss {best_val_loss}")
    # Load the best saved model.
    with open(model_path, "rb") as f:
        model = torch.load(f).to(device)
    return model
