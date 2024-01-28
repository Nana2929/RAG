# RAG
Retrieval-augmented Generation

- `retrieve.ipynb`: 從指定的資料集中抓出任一 user-item 組合，呈現他們的相關資料與 Multi-aspect PEPLER 的生成結果。
   - `#Custom Arguments` 可以更改資料集名稱，資料和路徑已經全部寫好（設定是在ikm9103上）。
   - `RAND_INDEX` 那個 cell 每次按都隨機抓一組 user-item 組合，可以用來觀察不同的組合的生成結果。
   - 由於 torch 內部存模型的方式有點彆扭，loading model 時要確保執行 runtime 內吃得到 `RecReg` 這個模型（`modeling_pepler_buildingblocks.py` 和 `modeling_pepler.py` 和執行檔案在同一資料夾），才能正確載入。
   - `yelp23`, `yelp` 資料量較大，載入時間較長。
   - `gest` 沒有 user 與 item 相關的 metadata，所以只會呈現 id 資訊不會有相關姓名地點等資訊。
