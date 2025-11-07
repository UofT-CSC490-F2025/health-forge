from train_rlvr_modal import app, run_rlvr  # import your Modal app + function

if __name__ == "__main__":
    with app.run():
        handle = run_rlvr.remote(
        real_s3_key="health-forge-data-processing/ehr_norm.npy",
        synth_s3_key="health-forge-data-processing/workdirs/judge_train/samples/all_x.npy",
        out_s3_prefix="health-forge-data-processing/rlvr_judge",
        epochs=10,
        lr=1e-5,
        kl_beta=0.01
        )

        # Stream logs live
        for log in handle.stream_logs():
            print(log, end="")

        # Wait for completion
        handle.wait()
