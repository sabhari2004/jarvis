import urllib.parse
import webbrowser

url = f"https://www.google.com/search?q={urllib.parse.quote('silent hill')}"
print("opening url...")
webbrowser.get().open(url)
print("opened!")
