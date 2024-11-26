import cv2
import mediapipe as mp
import numpy as np

def cut_out(image_path, output_path):
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    segmentation_model = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = segmentation_model.process(image_rgb)
    mask = results.segmentation_mask
    binary_mask = (mask > 0.5).astype(np.uint8)  # Binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(np.vstack(contours))  # Bounding box of the person
    print(f"Bounding box position: x={x}, y={y}, width={w}, height={h}")
    person_cutout = cv2.bitwise_and(image, image, mask=binary_mask)
    output_image = image.copy()
    cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(output_path, person_cutout)
    cv2.imwrite("output_with_bbox.jpg", output_image)
    print(f"Cut-out image saved at: {output_path}")
    print(f"Image with bounding box saved at: output_with_bbox.jpg")
    return [x,y,w,h]

def add_text(image_path, output_path, text, position, font_scale=1, color=(255, 255, 255), thickness=2):
    image = cv2.imread(image_path)
    # Define the font
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)
    cv2.imwrite(output_path, image)
    print(f"Text added to the image and saved at: {output_path}")

def paste_img(background_path, overlay_path, output_path, position=(0, 0)):
    # Load the background and overlay images
    background = cv2.imread(background_path)
    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    overlay_h, overlay_w = overlay.shape[:2]
    bg_h, bg_w = background.shape[:2]
    x, y = position
    if x + overlay_w > bg_w or y + overlay_h > bg_h:
        raise ValueError("Overlay image exceeds background boundaries. Adjust the position or resize the overlay.")
    roi = background[y:y + overlay_h, x:x + overlay_w]
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([10, 10, 10], dtype=np.uint8)
    mask = cv2.inRange(overlay, lower_black, upper_black)
    mask_inv = cv2.bitwise_not(mask)
    overlay_fg = cv2.bitwise_and(overlay, overlay, mask=mask_inv)
    background_bg = cv2.bitwise_and(roi, roi, mask=mask)
    combined = cv2.add(background_bg, overlay_fg)
    background[y:y + overlay_h, x:x + overlay_w] = combined
    cv2.imwrite(output_path, background)
    print(f"Image saved at: {output_path}")

# paste setting
# background_path = "outputtxt.png"  # Path to the background image
# overlay_path = "output_cutout.png"       # Path to the overlay image
# output_path = "output.jpg"         # Path to save the final image
position_cutout = (0,0)               # Top-left position for the overlay
# alpha =0.81                  # Transparency of the overlay

# text settings
text = "PYTHON"  # Text to add
position = (3, 17+225)  # Position (x, y) for the text
font_scale = 3 
color = (0, 255, 0)  # Green text
thickness = 3  

# image settings
orignial_img = "files/image.png"  # Path to the input image
cut_out_img = "outputtxt.png"  # Output image path
text_img = "output_txt.png"
final_output = "final_output.png"

cut_out(orignial_img, cut_out_img)
add_text(orignial_img, text_img, text, position, font_scale, color, thickness)
paste_img(text_img, cut_out_img, final_output, position_cutout)
