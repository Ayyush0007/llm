import requests
import sys

def test_inference(image_path, api_url="https://sarasproject-bento.hf.space/predict"):
    print(f"🚀 Testing inference for: {image_path}")
    print(f"🌐 API URL: {api_url}")
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = requests.post(api_url, files=files)
        
        if response.status_code == 200:
            data = response.json()
            detections = data.get("detections", [])
            print(f"✅ Success! Found {len(detections)} detections.")
            for det in detections:
                print(f"  - {det['class']}: {det['confidence']:.2f}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"💥 Failed to connect: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # Default test image from the dataset
        path = "/Users/yashmogare/robot-ai/llm/robot-ai/idd-detection1-1/test/images/20200619_162335_jpg.rf.e11ebf5139a03e3b886d41520a2f1317.jpg"
    
    test_inference(path)
