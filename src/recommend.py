"""নিয়মভিত্তিক স্বাস্থ্য পরামর্শ (বাংলা + ইংরেজি)।

রোগীর ইনপুট মান বিশ্লেষণ করে সাধারণ, শিক্ষামূলক স্বাস্থ্য পরামর্শ দেয়।
এটি চিকিৎসা পরামর্শ নয় — শুধুমাত্র সচেতনতামূলক তথ্য।
প্রতিটি পরামর্শ {"bn": ..., "en": ...} আকারে রিটার্ন হয়।
"""

from __future__ import annotations


def get_recommendations(features: dict) -> list[dict]:
    """ইনপুট অনুযায়ী প্রাসঙ্গিক দ্বিভাষিক স্বাস্থ্য পরামর্শের তালিকা।"""
    tips: list[dict] = []

    glucose = float(features.get("glucose", 0))
    bmi = float(features.get("bmi", 0))
    bp = float(features.get("blood_pressure", 0))
    chol = float(features.get("cholesterol", 0))
    smoking = float(features.get("smoking", 0))
    activity = float(features.get("physical_activity", 0))
    heart_rate = float(features.get("heart_rate", 0))

    if glucose >= 140:
        tips.append({"bn": "🩸 রক্তে গ্লুকোজ বেশি — চিনি ও পরিশোধিত শর্করা কমান এবং নিয়মিত গ্লুকোজ পরীক্ষা করুন।",
                     "en": "🩸 High blood glucose — cut sugar and refined carbs, and test glucose regularly."})
    elif glucose >= 110:
        tips.append({"bn": "🩸 গ্লুকোজ সীমার কাছাকাছি — সুষম খাদ্য ও পরিমিত শর্করা বজায় রাখুন।",
                     "en": "🩸 Glucose near borderline — keep a balanced diet with moderate carbs."})

    if bmi >= 30:
        tips.append({"bn": "⚖️ বিএমআই স্থূলতার পর্যায়ে — ওজন কমাতে খাদ্য নিয়ন্ত্রণ ও ব্যায়াম শুরু করুন।",
                     "en": "⚖️ BMI in obese range — start diet control and exercise to lose weight."})
    elif bmi >= 25:
        tips.append({"bn": "⚖️ বিএমআই কিছুটা বেশি — স্বাস্থ্যকর ওজনের জন্য সক্রিয় জীবনযাপন রাখুন।",
                     "en": "⚖️ BMI slightly high — stay active for a healthy weight."})
    elif 0 < bmi < 18.5:
        tips.append({"bn": "⚖️ বিএমআই কম — পুষ্টিকর খাবার বাড়ান।",
                     "en": "⚖️ BMI low — increase nutritious food intake."})

    if bp >= 130:
        tips.append({"bn": "❤️ রক্তচাপ বেশি — লবণ কমান, মানসিক চাপ নিয়ন্ত্রণ করুন এবং নিয়মিত রক্তচাপ মাপুন।",
                     "en": "❤️ High blood pressure — reduce salt, manage stress, and monitor BP regularly."})

    if chol >= 240:
        tips.append({"bn": "🧈 কোলেস্টেরল উচ্চ — চর্বিযুক্ত খাবার এড়িয়ে আঁশযুক্ত খাবার বাড়ান।",
                     "en": "🧈 High cholesterol — avoid fatty food and eat more fiber."})

    if smoking >= 1:
        tips.append({"bn": "🚭 ধূমপান হৃদরোগ ও ডায়াবেটিসের ঝুঁকি বাড়ায় — ধূমপান ত্যাগের পরিকল্পনা করুন।",
                     "en": "🚭 Smoking raises heart disease and diabetes risk — plan to quit."})

    if activity <= 3:
        tips.append({"bn": "🏃 শারীরিক সক্রিয়তা কম — সপ্তাহে অন্তত ১৫০ মিনিট মাঝারি ব্যায়াম করুন।",
                     "en": "🏃 Low physical activity — aim for at least 150 minutes of moderate exercise weekly."})

    if heart_rate >= 100:
        tips.append({"bn": "💓 বিশ্রামকালীন হৃদস্পন্দন বেশি — পর্যাপ্ত বিশ্রাম ও চিকিৎসকের পরামর্শ নিন।",
                     "en": "💓 High resting heart rate — get adequate rest and consult a doctor."})

    if not tips:
        tips.append({"bn": "✅ আপনার সূচকগুলো মোটামুটি স্বাভাবিক — সুষম খাদ্য ও নিয়মিত ব্যায়াম বজায় রাখুন।",
                     "en": "✅ Your indicators look largely normal — maintain a balanced diet and regular exercise."})

    return tips
