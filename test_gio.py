from gi.repository import Gio
import urllib.parse
url = f"https://www.google.com/search?q={urllib.parse.quote('silent hill')}"
app = Gio.AppInfo.get_default_for_uri_scheme("https")
print("Gio app is", app.get_name() if app else "None")
if app:
    app.launch_default_for_uri(url)
print("launched!")
