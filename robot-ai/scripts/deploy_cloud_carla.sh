#!/bin/bash
# deploy_cloud_carla.sh — 1-Click CARLA Cloud Deployment
# Run this on a Cloud GPU instance (e.g. AWS EC2, RunPod, GCP, Lambda Labs)
# Requires an NVIDIA GPU and Ubuntu Linux.

echo "==============================================="
echo "🏎️  Deploying CARLA Headless Server on Cloud ☁️"
echo "==============================================="

# 1. Update and install dependencies
echo ">> Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y wget unzip xdg-user-dirs screen

# 2. Download CARLA 0.9.15 (Latest stable for UE4)
echo ">> Downloading CARLA 0.9.15 (this might take a few minutes)..."
cd ~
wget -q -c https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz

# 3. Extract CARLA
echo ">> Extracting CARLA..."
mkdir -p carla_server
tar -xzf CARLA_0.9.15.tar.gz -C carla_server/

# 4. Create a startup script for headless mode (No GUI, rendering offscreen)
echo ">> Creating headless startup script..."
cat << 'EOF' > carla_server/start_headless.sh
#!/bin/bash
# Runs CARLA off-screen, rendering using the GPU via Vulkan/OpenGL
./CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Low -carla-rpc-port=2000
EOF
chmod +x carla_server/start_headless.sh

# 5. Start CARLA in a detached screen session
echo ">> Starting CARLA Server in background screen (named 'carla')..."
screen -dmS carla bash -c "cd ~/carla_server && ./start_headless.sh"

echo "==============================================="
echo "✅ CARLA is now running in the background!"
echo ""
echo "To view server logs, run: screen -r carla"
echo "(Press Ctrl+A, then D to detach from the screen without killing it)"
echo ""
echo "🔥 NEXT STEP ON YOUR MAC:"
echo "Run this command on your Mac to tunnel the connection so your Python code thinks CARLA is running locally:"
echo ""
echo "  ssh -N -L 2000:localhost:2000 -L 2001:localhost:2001 -L 2002:localhost:2002 ubuntu@YOUR_CLOUD_IP"
echo ""
echo "==============================================="
