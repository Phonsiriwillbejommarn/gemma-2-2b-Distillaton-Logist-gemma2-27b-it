import os
from huggingface_hub import snapshot_download, login

# --- กำหนดค่าตรงนี้ ---
HF_TOKEN = "API_KEY"                      # ใส่ HF Token ของคุณตรงนี้
MODEL_ID = "Phonsiri/gemma-2-2b-SFT-Reasoning-full-Model"  # ชื่อ Model บน Hugging Face ของคุณ
LOCAL_DIR = "./sft_output"                    # โฟลเดอร์ปลายทางที่จะบันทึกบนเครื่อง

def main():
    print(f"กำลังเริ่มดาวน์โหลดโมเดล {MODEL_ID}...")
    
    # 1. ล็อกอินเข้า Hugging Face 
    if HF_TOKEN and HF_TOKEN != "API_KEY":
        print("กำลังล็อกอินเข้าสู่ Hugging Face...")
        login(token=HF_TOKEN)
    else:
        print("คำเตือน: ไม่พบ HF_TOKEN ถ้าโมเดลตั้งเป็น Private การโหลดจะล้มเหลวได้")

    # 2. จำลองการสร้างโฟลเดอร์ถ้ายังไม่มี
    os.makedirs(LOCAL_DIR, exist_ok=True)
    print(f"บันทึกไฟล์ไปที่: {LOCAL_DIR}")

    # 3. ดาวน์โหลด
    try:
        download_path = snapshot_download(
            repo_id=MODEL_ID,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False,  # ให้มาเป็นไฟล์จริงๆ ไม่ใช่แค่ลิ้งค์
            resume_download=True           # โหลดต่อได้ถ้าเน็ตหลุด
        )
        print("\n✅ ดาวน์โหลดสำเร็จเรียบร้อย!")
        print(f"ไฟล์ถูกจัดเก็บพร้อมใช้งานสำหรับรัน Distillation ต่อที่: {download_path}")
    except Exception as e:
        print(f"\n❌ เกิดข้อผิดพลาดระหว่างดาวน์โหลด: {e}")

if __name__ == "__main__":
    main()
