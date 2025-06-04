import modal

# Define an image that includes fastapi and its dependencies
image = modal.Image.debian_slim().pip_install("fastapi[standard]")

# Define the Modal app, specifying the image for functions by default if desired,
# or specify per function.
app = modal.App("tabrl-hello") 

@app.function(image=image)  # Apply the image to this specific function
@modal.fastapi_endpoint(method="POST")
def hello():
    return {"message": "Hello from Modal!", "status": "connected"}

# Test locally: modal serve hello_world.py
# Deploy: modal deploy hello_world.py
# (After running `modal serve`, you can test with a POST request, e.g., using curl:
# curl -X POST http://localhost:8000 # (Port might vary, check serve output))

