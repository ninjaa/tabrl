# TabRL - Train Robots in Your Browser Tab

Everything runs in one browser tab except LLM calls and GPU training.

## Quick Start

1.  **Clone robot scenes:**
    ```bash
    ./scripts/setup_scenes.sh
    ```

2.  **Set up Python Environment (using `uv`):**
    *   Create the virtual environment:
        ```bash
        uv venv .venv
        ```
    *   Activate the environment:
        ```bash
        source .venv/bin/activate
        ```
    *(Ensure `uv` is installed. Alternatively, use `python -m venv .venv` and then `source .venv/bin/activate`)*

3.  **Install Modal:**
    ```bash
    uv pip install modal
    ```
    *(If not using `uv`, ensure your virtual environment is active and run `pip install modal`)*

4.  **Configure Modal CLI:**
    ```bash
    modal setup
    ```

5.  **Install Project Dependencies (e.g., for frontend):**
    ```bash
    npm install
    ```

6.  **Run Development Server:**
    ```bash
    npm run dev
    ```
