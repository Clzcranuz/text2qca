"""Streamlit Community Cloud entry point — proxies to app.py."""
import runpy
import pathlib

runpy.run_path(str(pathlib.Path(__file__).parent / "app.py"), run_name="__main__")
