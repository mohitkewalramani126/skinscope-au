from vision.inference import analyze_image

with open("test_images/IMG_2084.jpg", "rb") as f:
    img_bytes = f.read()

result = analyze_image(img_bytes)
print("risk_score:", result.risk_score)
print("risk_band:", result.risk_band)
print("raw_logit:", result.raw_logit)
print("mask base64 length:", len(result.mask_png_base64))