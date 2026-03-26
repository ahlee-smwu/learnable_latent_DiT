import random
import shutil
import tempfile
from pathlib import Path
from cleanfid import fid

def sample_two_non_overlapping_sets(
    src_dir: str,
    n_each: int = 20000,
    seed: int = 42,
) -> None:
    src_path = Path(src_dir)

    if not src_path.is_dir():
        raise ValueError(f"Source directory does not exist: {src_path}")

    # 현재 폴더 바로 아래의 .webp 파일만 사용
    webp_files = list(src_path.glob("*.webp"))

    total_needed = n_each * 2
    if len(webp_files) < total_needed:
        raise ValueError(
            f"Not enough .webp files. "
            f"Found {len(webp_files)}, but need at least {total_needed}."
        )

    rng = random.Random(seed)

    # 겹치지 않도록 총 40000개를 한 번에 샘플링
    selected_files = rng.sample(webp_files, total_needed)
    sample1_files = selected_files[:n_each]
    sample2_files = selected_files[n_each:]

    # 임시 폴더 생성: with 블록이 끝나면 자동 삭제
    with tempfile.TemporaryDirectory(prefix="webp_samples_") as tmp_root:
        tmp_root_path = Path(tmp_root)
        sample1_dir = tmp_root_path / "sample1"
        sample2_dir = tmp_root_path / "sample2"

        sample1_dir.mkdir(parents=True, exist_ok=True)
        sample2_dir.mkdir(parents=True, exist_ok=True)

        # 파일 복사
        for file_path in sample1_files:
            shutil.copy2(file_path, sample1_dir / file_path.name)

        for file_path in sample2_files:
            shutil.copy2(file_path, sample2_dir / file_path.name)

        print(f"Temporary root: {tmp_root_path}")
        print(f"Sample 1 dir: {sample1_dir} ({len(sample1_files)} files)")
        print(f"Sample 2 dir: {sample2_dir} ({len(sample2_files)} files)")
        print("두 폴더는 서로 겹치지 않습니다.")

        fid_all = fid.compute_fid(
            fdir1=str(sample1_dir),
            fdir2=str(sample2_dir),
            mode="clean",
            num_workers=8,
            batch_size=32
        )
        print(f"[FID] ALL clusters: {fid_all:.4f}")

        input("작업이 끝나면 Enter를 누르세요...")

    # with 종료 시 tmp_root 전체 자동 삭제
    print("임시 폴더가 삭제되었습니다.")


if __name__ == "__main__":
    sample_two_non_overlapping_sets("/mnt/HDD2/dataset/lsun/church_outdoor_train/church", n_each=20000, seed=42)