from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from collections import defaultdict
import visualkeras
from PIL import ImageFont
import tensorflow_addons as tfa

def load_font(font_path="arial.ttf", font_size=450):  # Already increased to 450
    try:
        return ImageFont.truetype(font_path, font_size)
    except:
        print("⚠️ Could not load font. Using default.")
        return None

def get_color_map():
    color_map = defaultdict(dict)
    color_map[layers.Conv2D]['fill'] = '#0077b6'           # Deep blue
    color_map[layers.Conv2DTranspose]['fill'] = '#00b4d8'  # Aqua
    color_map[layers.MaxPooling2D]['fill'] = '#d00000'     # Strong red
    color_map[layers.AveragePooling2D]['fill'] = '#ff8800' # Orange
    color_map[layers.Dropout]['fill'] = '#6a4c93'          # Deep purple
    color_map[layers.Dense]['fill'] = '#2ec4b6'            # Teal
    color_map[layers.Flatten]['fill'] = '#ffb703'          # Gold
    color_map[layers.BatchNormalization]['fill'] = '#3c096c'  # Violet
    color_map[layers.InputLayer]['fill'] = '#80ffdb'       # Mint
    color_map[layers.Activation]['fill'] = '#fb5607'       # Bright orange
    color_map[layers.UpSampling2D]['fill'] = '#f15bb5'     # Pink
    color_map[layers.Concatenate]['fill'] = '#7209b7'      # Dark purple
    color_map[layers.Add]['fill'] = '#2a9d8f'              # Teal green (added for DIES model)
    color_map[layers.Lambda]['fill'] = '#f8f9fa'           # Light gray (for median filter)
    return color_map

def visualize_model(model, output_path=None, legend=True, spacing=50, font_size=450):
    # Increased font_size parameter default to 450
    font = load_font(font_size=font_size)
    color_map = get_color_map()

    # Using the larger font for the entire visualization
    image = visualkeras.layered_view(
        model,
        legend=legend,
        font=font,
        spacing=spacing,
        color_map=color_map
    )

    if output_path:
        image.save(output_path)
        print(f"✅ Diagram saved to: {output_path}")
    else:
        image.show()

# 🧪 Example usage
model = load_model("models/DIES-Deep-Image-Enhancing-Smoothing-Model/DIES_Model.h5")
visualize_model(
    model,
    output_path="dies_model_architecture.png",
    legend=True,
    spacing=70,        # Increased spacing between layers for better visibility
    font_size=80      # Already set to large font size
)