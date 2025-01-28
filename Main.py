import os
import argparse
from MaskDetector import MaskDetector

def main():
    parser = argparse.ArgumentParser(description="Detect if a person is wearing a mask in an image.")

    parser.add_argument(
        "--image_path",
        type=str,
        default=os.path.join('FaceMaskDataSet', 'test_image', 'with_mask_5.jpg'),
        help="Path to the input image."
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="mask_detector_model.h5",
        help="Path to the trained model file. Default is 'mask_detector_model.h5'."
    )

    args = parser.parse_args()
    detector = MaskDetector(model_path=args.model_path)

    try:
        result = detector.predict(args.image_path)
        print(f"\n Prediction: {result}\n")
    except ValueError as e:
        raise RuntimeError(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
