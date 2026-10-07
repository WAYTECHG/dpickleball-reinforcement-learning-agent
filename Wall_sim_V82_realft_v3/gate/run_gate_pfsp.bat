@echo off
cd /d D:\pickleball\WW_SelfPlay_HR
"D:\ProgramData\Anaconda3\envs\dpickleball\python.exe" batch_gate.py ^
 --candidates "D:\pickleball\Smash_Potato_HR_v1_LOCKED\model_left.pt,D:\pickleball\Wall_sim_V82_realft\checkpoints\realft_00700004_policy.pt,D:\pickleball\Wall_sim_V82_realft\checkpoints\realft_00900004_policy.pt,D:\pickleball\Wall_sim_V82_realft\checkpoints\realft_01100004_policy.pt,D:\pickleball\Wall_sim_V82_realft\checkpoints\realft_01350004_policy.pt,D:\pickleball\Wall_sim_V82_realft\checkpoints\realft_01600004_policy.pt" ^
 --opponent "D:\pickleball\Wall_sim_V67_FT\pretrained\v54_v67_originals\best_2_policy.pt" ^
 --env-path "D:\pickleball\dPickleball BuildFiles\Competition\Windows\dp.exe" ^
 --steps 40000 --target-score 21 --base-worker-id 80 ^
 --out-dir "logs\batch_gate_pfsp" > "logs\batch_gate_pfsp_console.log" 2>&1
