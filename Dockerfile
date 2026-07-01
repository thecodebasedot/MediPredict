FROM python:3.11-slim

WORKDIR /app

# নির্ভরতা ইনস্টল (ক্যাশ সুবিধার জন্য আগে কপি)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# সোর্স কপি
COPY . .

# মডেল তৈরি করা (ইমেজ বিল্ড টাইমে)
RUN python -m src.train

EXPOSE 5000

# হেলথচেক — /api/health এ পোল করা হয়
HEALTHCHECK --interval=30s --timeout=4s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/api/health')" || exit 1

# প্রোডাকশন WSGI সার্ভার (gunicorn) দিয়ে চালু
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "60", "app.app:app"]
