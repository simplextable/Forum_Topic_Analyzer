import pandas as pd

# CSV dosyasını oku (virgül karışıklığı olmaması için quotechar="\"" önerilir)
df = pd.read_csv("sentiment_results.csv", quotechar='"')

# 'sentiment' sütununu en başa al
columns = ['sentiment'] + [col for col in df.columns if col != 'sentiment']
df = df[columns]

# Excel'e kaydet
df.to_excel("sentiment_results.xlsx", index=False)
