
## Create a New Virtual Environment:

To manage separate package installations for different projects, use venv (for Python 3) to create a virtual environment. This isolates your Python installation for each project, ensuring that packages do not interfere with each other.

Tip: Always use virtual environments when working with third-party packages.

In your project's directory, run the following command to create a virtual environment named .venv:

### Unix/macOS:

```bash
python3 -m venv .venv
```
### Windows:

```bash
python -m venv .venv
```
Note: Exclude the virtual environment directory from version control using .gitignore.

## Activate a Virtual Environment:

Before installing or using packages in the virtual environment, you need to activate it. This puts the virtual environment-specific Python and Pip executables into your shell's PATH.

### Unix/macOS:

```bash
source .venv/bin/activate
```
### Windows:

```bash
.venv\Scripts\activate
```
To confirm activation, check the location of your Python interpreter:

### Unix/macOS:

```bash
which python
```
### Windows:

```bash
where python
```

While the virtual environment is active, the output should include the .venv directory, confirming that Pip will install packages into that specific environment. This allows you to import and use packages in your Python application.


## Prepare Pip
### Install or Upgrade Pip:

Pip is the official Python package manager used to install and update packages in a virtual environment. Ensure that pip is up-to-date by running the following commands:

### Unix/macOS:

```bash
python3 -m pip install --upgrade pip
python3 -m pip --version
```


## Install Packages from requirements.txt:

Following command reads the specifications from the requirements.txt file and installs the specified packages in your virtual environment

```bash
pip install -r requirements.txt
```


## Configuring WebSocket Connection with API Token

To set up the WebSocket connection in whisper.py file, follow the example below:

```bash
url=LIVEKIT_URL
token = (
    api.AccessToken(api_key=API_KEY, api_secret=API_SECRET)
    .with_identity("KITT")
    .with_name("KITT")
    .with_grants(api.VideoGrants(room_join=True, room=ROOM_NAME))
    .to_jwt()
)
```

