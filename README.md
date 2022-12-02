# pgm-project

## To run for pong dataset:
python train.py --data_dir data/interventional_pong  --lambda_reg 0 --max_epochs 200

For quick experimentation, set max_epochs to 1. Otherwise, to check you're getting correct results, set max_epochs to 200

Once your code has run, you should have a new checkpoints folder. To view the results in tensorboard, do: 

tensorboard --logdir pgm-project/checkpoints/CITRISVAE/CITRISVAE_16l_5b_32hid_pong/(YOUR_VERSION) --port 6006

Each run creates a new version, so make sure you're using the latest one to view the results.

To check you're getting the correct results after 200 epochs, compare with the results from the project update report. 
