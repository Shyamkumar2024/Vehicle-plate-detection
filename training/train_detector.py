import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["vehicle", "plate"], required=True)
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    data_yaml = "training/vehicle_data.yaml" if args.task == "vehicle" else "training/plate_data.yaml"
    model = YOLO(args.model)
    model.train(data=data_yaml, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                patience=10, project="outputs/training", name=f"{args.task}_detector")
    best = Path("outputs/training") / f"{args.task}_detector" / "weights" / "best.pt"
    if best.exists():
        target = Path("models") / ("vehicle_detector.pt" if args.task == "vehicle" else "plate_detector.pt")
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(best.read_bytes())
        print(f"Copied best checkpoint to {target}")
    else:
        print("Training finished but expected best.pt was not found; inspect outputs/training.")

if __name__ == "__main__":
    main()
