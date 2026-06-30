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

# Flask অ্যাপ চালু — কন্টেইনারে 0.0.0.0 এ bind করা হয়
ENV FLASK_RUN_HOST=0.0.0.0
CMD ["python", "-c", "from app.app import app; app.run(host='0.0.0.0', port=5000)"]
