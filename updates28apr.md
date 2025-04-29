**project will be runnable now, hopefully**

1.  **Dependencies:** You need to have the conda environment (`environment.yml`) or pip requirements installed.
2.  **Data:** You need a text file specified via the `--data` argument (e.g., `text_data.txt`).
3.  **Execution:** You can now run specific stages:
    *   **To train the autoencoder (requires learnable encoder/decoder):**
        ```bash
        python train.py --data your_data.txt --stage reconstruction --encoder_type learnable --decoder_type learnable --vocab_size 500 --num_epochs 20 --save_dir checkpoints_ae
        ```
    *   **To train the diffusion model (loading pre-trained AE):**
        ```bash
        # Assuming the AE checkpoint is saved at checkpoints_ae/reconstruction/model_best.pt
        python train.py --data your_data.txt --stage diffusion --encoder_type learnable --decoder_type learnable --diffusion_type advanced --load_checkpoint checkpoints_ae/reconstruction/model_best.pt --num_epochs 50 --save_dir checkpoints_diffusion
        ```
    *   **To train diffusion with a deterministic setup (no pre-training needed):**
        ```bash
        python train.py --data your_data.txt --stage diffusion --encoder_type deterministic --decoder_type deterministic --diffusion_type simple --num_epochs 30 --save_dir checkpoints_det_simple
        ```
4.  **Gradient Issue (Reminder):** The `reconstruction` stage still uses the simple sampling method, which means the **encoder won't train effectively** due to the lack of gradient flow. The decoder *will* train. To make the encoder train properly in this stage, the **Gumbel-Softmax** change (discussed in the TODO comment in `model.py`) is still necessary. However, the *code structure* is now in place to run these stages separately.

The script will *run* and execute the selected stage's logic. The *effectiveness* of the 'reconstruction' stage for training the encoder is limited until the sampling method is improved. The 'diffusion' stage should train correctly based on the targets provided by the (potentially fixed or pre-trained) encoder.
