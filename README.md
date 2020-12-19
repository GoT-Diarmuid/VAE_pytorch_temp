# VAE_pytorch_temp
forTemporayTest

ml-20m 連結: https://reurl.cc/3L0N30

放置位置可見preprocessing_try.py ln.23

執行資料預處理(建議用jupyter分段生成):
python preprocessing_try.py

執行主程式:
python run_try.py

目前單純原本的架構把latent code跟origin decoder output接在一起放入attention，decoder還未增加層數.