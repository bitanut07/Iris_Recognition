export MONGODB_URL="mongodb+srv://anhtuant1234_db_user:fGQb9VniIVVwrxFm@cluster0.mtxt3pa.mongodb.net/"
python -m src.app.demo \
    --seg-ckpt checkpoints/mmu/mmu_epoch10.pth \
    --rec-ckpt checkpoints/recognition/recognition_epoch_11.pth \
    --metric cosine \
    --threshold 0.7