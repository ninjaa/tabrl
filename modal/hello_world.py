import modal

app = modal.App("tabrl-hello")

@app.function()
@modal.web_endpoint(method="POST")
def hello():
    return {"message": "Hello from Modal!", "status": "connected"}

# Test locally: modal serve hello_world.py
# Deploy: modal deploy hello_world.py
