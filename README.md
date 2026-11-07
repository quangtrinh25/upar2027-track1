# UPAR Challenge 2027, Track 1 (Pedestrian Attribute Recognition): code dg_v1

Gói này chỉ chứa code của Track 1: train, chọn ensemble và đóng gói bài nộp. Không kèm dữ liệu, weights pretrained,
checkpoint hay cache. Một số module dùng chung (`ensemble.py`, `dg_runtime.py`, `scoring.py`) còn nhánh Track 2, có thể bỏ qua.

## Kết quả

Bài nộp T1 tốt nhất hiện tại là **C6F2: Avg .7826 trên test ẩn** (mA .7888, Instance-F1 .7766, Prec .7539, Rec .8145).

| Bài nộp | Thay đổi chính | Avg test ẩn |
|---|---|---|
| Arm A | Recipe dg_v1: partition trung thực, Tiny + DINOv2-B + ConvNeXtV2-B | .7537 |
| LA | Thêm prior logit adjustment (τ=.6) | .7704 |
| L5 | Thêm DINOv2-L | .7782 |
| C6F2 | Thêm CLIP-L, model fp16, chọn theo giới hạn kích thước | **.7826** |

Mức tăng trên public val chuyển sang test ẩn gần 1:1 qua các lượt (ví dụ C6F2: val +.0033, test +.0044).

Cấu hình C6F2 nằm ở `reference/selection_track1_C6F2.json` (trọng số, τ, bias) và `reference/inference_track1_C6F2.json`
(file `assets/inference.json` trong ZIP đã nộp).
- Thành phần: DINOv2-L .4219, CLIP-L .25, ConvNeXt-Tiny BCE .1875, ConvNeXtV2-B .1406.
- τ=.6. Tune HM .9215, public val .9006 (×2 downscale: .8953).

## Phương pháp (dg_v1)

1. **Partition trung thực** (`src/upar/dg/partition.py`). Gộp ảnh gần trùng trong cùng nguồn bằng union-find (thumbnail
   cos > .9), rồi kiểm rò rỉ fit→tune bằng feature ConvNeXt-T ImageNet. PA100k không thể chia theo danh tính, nên nếu
   tỉ lệ cặp gần trùng vượt 2 lần nhóm đối chứng val→train (ngưỡng p99,9) thì chia lại một lần. Tune khoảng 10%.
2. **Recipe train chung** (`src/upar/dg/training.py`, `models.py`):
   - Augmentation `ema_aug` (flip, sáng/tương phản ±10%, downscale, blur, JPEG, scale/dịch nhỏ), không đổi màu sắc (hue).
   - BCE + label smoothing .02; drop path; weight decay .05, trừ norm, bias và embedding.
   - AdamW, 12 epoch (warmup 2 + cosine), AMP, batch hiệu dụng 64.
   - Backbone (weights pretrained tổng quát, không pretrain cho PAR): ConvNeXt-Tiny (torchvision),
     ConvNeXtV2-B in22k, DINOv2-B/L (reg4), CLIP ViT-L/14-336, SigLIP-L/16-256 (timm).
     ViT dùng input 252×126 (bội số 14), còn lại 256×128. CLIP/SigLIP tự chuẩn hóa lại input bên trong model.
   - Chọn checkpoint T1 theo HM(mA, Instance-F1) trên tune, ghi lại theo từng epoch.
3. **Chọn ensemble** (`src/upar/dg/ensemble.py: select_la`):
   - Prior logit adjustment: bias = −τ·logit(tỉ lệ dương trên tập fit), τ ∈ [.30, .70] chọn trên tune.
     Bước này kéo recall của thuộc tính hiếm lên (+.017 trên test ẩn).
   - Trọng số chọn bằng greedy mixing trên tune, trong giới hạn kích thước (`--budget-mb`) hoặc compute (`--budget-gflops`).
     Mọi tổ hợp tối đại đều được thử, để tránh greedy kẹt.
   - Public val chỉ dùng một lần làm gate quyết định nộp (`gate`).
4. **Đóng gói** (`src/upar/dg/package.py`, `dg_runtime.py`):
   - Mỗi thành phần được trace TorchScript fp16 (Torch 2.4.1, đúng image chấm), vào/ra float32, kiểm lệch < 1e-3.
   - Runtime đọc mỗi ảnh một lần, tiền xử lý một lần cho mỗi kích thước input, và giải mã lô sau trong lúc GPU chạy lô trước.
   - `BATCH_SIZE = 65536`, nên ingestion gọi `predict_batch` một lần cho cả tập.
   - Kèm smoke test và kiểm parity CPU so với cache GPU.

## Ràng buộc của cuộc thi

- Runner: A10 24 GB, **5 phút mỗi lượt chạy**, tối đa **3 GB** mỗi track.
  - C6F2 (226 GFLOPs/ảnh) chạy hết 266 giây với runtime cũ.
  - Runtime mới đạt khoảng 183 ảnh/giây với 205 GFLOPs/ảnh (đo trên Track 2), nên còn dư cho khoảng 280 GFLOPs.
  - FLOPs từng model: `GFLOPS` trong `ensemble.py`.
- **Giá trị T1 trên leaderboard là trung bình giữa các dataset test**, mỗi dataset nặng như nhau. Ngưỡng là 0,5.
- Leaderboard hiển thị **lượt nộp mới nhất**, không phải lượt tốt nhất.
- Không dùng ảnh test để thích nghi (không ước lượng prior bằng EM, không hiệu chỉnh theo thống kê tập test, không chỉnh BatchNorm trên test),
  và không dùng cấu trúc của tập test ẩn để chọn bất cứ thứ gì.

## Cài đặt

- Môi trường train và chọn: `requirements-train.txt` (Torch 2.11 CUDA, timm 1.0.22).
- Môi trường export và đóng gói: `requirements-export.txt` (**Torch 2.4.1**, đúng image chấm). Smoke test của `package`
  gọi `.venv-compat/bin/python` (Torch 2.4.1), nên cần tạo venv đó hoặc sửa đường dẫn trong `scripts/dg_v1.py: cmd_package`.
- Test: `python -m pytest -q tests/test_dg_partition.py tests/test_dg_training.py tests/test_dg_ensemble.py tests/test_dg_v5.py tests/test_metrics.py tests/test_scoring.py`
  (31 test, môi trường train) và `tests/test_dg_package.py` (môi trường export). Đã chạy pass trong một bản sao sạch của gói này.

## Dữ liệu và đường dẫn cần sửa

| Nội dung | Vị trí mặc định |
|---|---|
| Annotation chính thức | `external/UPAR-Challenge-2027/data/annotations/` (kèm `peta_file_mapping.txt`) |
| Ảnh | `data/<đường dẫn trong annotation>` |
| Audit trùng lặp | `reports/data_audit.json` (có sẵn trong gói) |
| Thư mục artifact campaign | hằng `CAMPAIGN` trong `scripts/dg_v1.py`, `scripts/dg_v1_status.py` |
| Weights DINOv2-B, ConvNeXtV2-B | `data/pretrained/competition_upgrade/{B,A}.safetensors` |
| Weights DINOv2-L, CLIP-L, SigLIP-L | đường dẫn tuyệt đối trong `src/upar/dg/models.py: SPECS` |

Weights lấy từ timm/Hugging Face theo `timm_name` trong `SPECS`, lưu dạng safetensors kèm file `.json` ghi nguồn gốc;
`_provenance` kiểm file này trước khi nạp.

Mọi job train, predict và đóng gói chạy qua ledger ngân sách GPU (`competition_next/resources.py: campaign_job`):
- Cần `<CAMPAIGN>/reports/protocol.json`: chép từ `reference/protocol_template.json` rồi sửa `artifact_root`.
- Mỗi job ghi `*_started.json` / `*_journal.json` và giữ khóa GPU.
- Job hỏng vẫn bị tính giờ. Chạy lại thì dùng `--attempt N`.

## Chạy lại C6F2

```bash
export PYTHONPATH=src
python scripts/dg_v1.py partition                       # partition A (fit/tune) và B
python scripts/dg_v1.py leakage                         # kiểm rò rỉ PA100k, chia lại tối đa một lần (GPU)
for arm in tiny_bce tiny_weighted dinov2_b convnextv2_b dinov2_l clip_l; do
  python scripts/dg_v1.py smoke --arm $arm               # kiểm VRAM/thông lượng (tuỳ chọn)
  python scripts/dg_v1.py train --arm $arm --data A
done
python scripts/dg_v1.py predict --data A --arms tiny_bce tiny_weighted dinov2_b convnextv2_b dinov2_l clip_l
python scripts/dg_v1.py select-la --name C6F2 --budget-mb 1450 \
  --arms tiny_bce tiny_weighted dinov2_b convnextv2_b dinov2_l clip_l
python scripts/dg_v1.py gate --name C6F2 --reference <selection đã nộp trước đó>
<venv-export>/bin/python scripts/dg_v1.py package --track 1 --data A --name C6F2
python scripts/dg_v1_benchmark.py <zip> --track 1 --batch 1024   # ước lượng thời gian trên GPU local
```

Mỗi model L mất khoảng 3,5–4 giờ trên GPU 16 GB (15–20 phút mỗi epoch). Các launcher thật đã dùng nằm trong `scripts/run_dg_v1_*.sh`.

## Đang làm (19/10)

- **dg_v4 (LLRD):** LR phân tầng theo độ sâu, block cuối 3e-5, giảm dần về embedding (hệ số .8 cho B, .9 cho L), arm `*_llrd`.
  Bản thử DINOv2-B so với LR phẳng: tune CA +.0019, mADM +.0047. CLIP-L và DINOv2-L LLRD đang train.
- **dg_v5:** LLRD cộng random erasing và loss softmax cạnh tranh trong nhóm thuộc tính chỉ có một đáp án
  (tuổi, tóc, màu áo, màu quần, kiểu quần), file `src/upar/dg/v5.py`. Đã có code và test, chưa train.
- **Đã thử, không giúp** (khỏi làm lại):
  - Train+val (Arm B); dg_v3 xóa dấu vết nguồn (méo tỉ lệ, độ phân giải, saturation, che nhãn).
  - Offset ngưỡng toàn cục; luật hậu xử lý màu (bắt buộc ít nhất 1 màu, chỉ giữ 1 màu, cạnh tranh có bù prior).
  - Ensemble 3 model L (+.0007 tune); chọn theo trung bình domain; distillation (trần thấp).
- Điểm yếu lớn nhất so với top 5: **recall thấp hơn .037 khi precision ngang nhau**. Màu sắc chiếm nhiều lỗi nhất.
  Thuộc tính yếu: Glasses-Sun, LowerBody-Color-Other, Age-Young, UpperBody-Color-Other, Glasses-Normal, Bald.
