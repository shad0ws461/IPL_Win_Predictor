import os
import requests

def download_file(url, dest_path):
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        print(f"Saved to {dest_path}")
    else:
        raise Exception(f"Failed to download from {url}. Status code: {response.status_code}")

def main():
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(workspace_dir, "data")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
        
    urls = {
        "matches.csv": "https://raw.githubusercontent.com/akashgupta4891/datasharing/master/matches.csv",
        "deliveries.csv": "https://raw.githubusercontent.com/akashgupta4891/datasharing/master/deliveries.csv"
    }
    
    for filename, url in urls.items():
        dest = os.path.join(data_dir, filename)
        try:
            download_file(url, dest)
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            
if __name__ == "__main__":
    main()
