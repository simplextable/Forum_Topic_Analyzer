from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import os
import json
import sys

def get_eksi_entries(baslik_url, max_entry=250):
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36")

    profile_dir = f"/tmp/chrome_user_data_{int(time.time())}"
    os.makedirs(profile_dir, exist_ok=True)
    options.add_argument(f"--user-data-dir={profile_dir}")

    driver = webdriver.Chrome(options=options)

    all_entries = []
    current_url = baslik_url
    page = 1

    while len(all_entries) < max_entry:
        driver.get(current_url)
        time.sleep(2)

        entry_divs = driver.find_elements(By.CLASS_NAME, "content")
        print(f"📄 {page}. sayfada {len(entry_divs)} entry bulundu.")

        for entry in entry_divs:
            if len(all_entries) >= max_entry:
                break
            all_entries.append(entry.text.strip())

        try:
            # "sonraki" butonu
            next_button = driver.find_element(By.CLASS_NAME, "next")
            next_url = next_button.get_attribute("href")
            if not next_url:
                break
            current_url = next_url
            page += 1
        except:
            break

    driver.quit()
    return all_entries

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Hata: Lütfen bir Ekşi Sözlük başlık URL'si girin.\nÖrnek: python eksi_scraper.py https://eksisozluk.com/baslik")
        sys.exit(1)

    url = sys.argv[1]
    results = get_eksi_entries(url, max_entry=250)

    print("\n🔍 Sonuçlar:")
    if not results:
        print("⚠️ Hiç entry bulunamadı.")
    else:
        for i, entry in enumerate(results, 1):
            print(f"{i}. {entry}\n{'-'*40}")

        with open("data/entries.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ {len(results)} entry 'entries.json' dosyasina yazildi.")






