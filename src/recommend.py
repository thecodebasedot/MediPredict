"""নিয়মভিত্তিক স্বাস্থ্য পরামর্শ (বাংলা)।

রোগীর ইনপুট মান বিশ্লেষণ করে সাধারণ, শিক্ষামূলক স্বাস্থ্য পরামর্শ দেয়।
এটি চিকিৎসা পরামর্শ নয় — শুধুমাত্র সচেতনতামূলক তথ্য।
"""

from __future__ import annotations


def get_recommendations(features: dict) -> list[str]:
    """ইনপুট অনুযায়ী প্রাসঙ্গিক স্বাস্থ্য পরামর্শের তালিকা রিটার্ন করে।"""
    tips: list[str] = []

    glucose = float(features.get("glucose", 0))
    bmi = float(features.get("bmi", 0))
    bp = float(features.get("blood_pressure", 0))
    chol = float(features.get("cholesterol", 0))
    smoking = float(features.get("smoking", 0))
    activity = float(features.get("physical_activity", 0))
    heart_rate = float(features.get("heart_rate", 0))

    if glucose >= 140:
        tips.append("🩸 রক্তে গ্লুকোজ বেশি — চিনি ও পরিশোধিত শর্করা কমান এবং নিয়মিত গ্লুকোজ পরীক্ষা করুন।")
    elif glucose >= 110:
        tips.append("🩸 গ্লুকোজ সীমার কাছাকাছি — সুষম খাদ্য ও পরিমিত শর্করা বজায় রাখুন।")

    if bmi >= 30:
        tips.append("⚖️ বিএমআই স্থূলতার পর্যায়ে — ওজন কমাতে খাদ্য নিয়ন্ত্রণ ও ব্যায়াম শুরু করুন।")
    elif bmi >= 25:
        tips.append("⚖️ বিএমআই কিছুটা বেশি — স্বাস্থ্যকর ওজনের জন্য সক্রিয় জীবনযাপন রাখুন।")
    elif bmi < 18.5 and bmi > 0:
        tips.append("⚖️ বিএমআই কম — পুষ্টিকর খাবার বাড়ান।")

    if bp >= 130:
        tips.append("❤️ রক্তচাপ বেশি — লবণ কমান, মানসিক চাপ নিয়ন্ত্রণ করুন এবং নিয়মিত রক্তচাপ মাপুন।")

    if chol >= 240:
        tips.append("🧈 কোলেস্টেরল উচ্চ — চর্বিযুক্ত খাবার এড়িয়ে আঁশযুক্ত খাবার বাড়ান।")

    if smoking >= 1:
        tips.append("🚭 ধূমপান হৃদরোগ ও ডায়াবেটিসের ঝুঁকি বাড়ায় — ধূমপান ত্যাগের পরিকল্পনা করুন।")

    if activity <= 3:
        tips.append("🏃 শারীরিক সক্রিয়তা কম — সপ্তাহে অন্তত ১৫০ মিনিট মাঝারি ব্যায়াম করুন।")

    if heart_rate >= 100:
        tips.append("💓 বিশ্রামকালীন হৃদস্পন্দন বেশি — পর্যাপ্ত বিশ্রাম ও চিকিৎসকের পরামর্শ নিন।")

    if not tips:
        tips.append("✅ আপনার সূচকগুলো মোটামুটি স্বাভাবিক — সুষম খাদ্য ও নিয়মিত ব্যায়াম বজায় রাখুন।")

    return tips
