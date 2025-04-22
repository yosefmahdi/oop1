import os
import re
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import google.generativeai as genai
from flask_cors import CORS
import tiktoken
from datetime import datetime
from docx import Document
import requests
import json
import PyPDF2
import pandas as pd
from bs4 import BeautifulSoup
import csv

# إعداد Flask
app = Flask(__name__)
CORS(app)

# التحقق من نوع الملف
valid_image_extensions = ['.png', '.jpg', '.jpeg']
valid_image_extensions_file_type = ['png', 'jpg', 'jpeg']
valid_doc_extensions = [
    # ملفات النصوص ومستندات Office
    '.txt', '.docx', '.docs', '.pdf', '.csv', '.xlsx', '.html', '.json',

    # قواعد البيانات
    '.sql', '.sqlite', '.db', '.bson', '.cql', '.neo4j',

    # ملفات الإعدادات والتكوين
    '.xml', '.yaml', '.yml', '.ini', '.conf', '.cfg',

    # ملفات البرمجة
    '.py', '.js', '.java', '.cpp', '.c', '.rb', '.php', '.ts', '.swift', '.go',
    '.cs', '.vb', '.scala', '.kt', '.rs', '.r', '.jl', '.pl', '.sh', '.bat', '.asm',
    '.lua', '.dart', '.erl', '.exs', '.ml', '.clj', '.fs', '.groovy', '.ps1', '.m',
    '.sas', '.sps', '.do', '.nb', '.tcl', '.ahk', '.applescript', '.vbs', '.tex',
    '.md', '.org', '.hs', '.adb', '.ads', '.for', '.f', '.f90',

    # ملفات التصميم
    '.css', '.scss', '.sass', '.less', '.styl'
]

valid_doc_extensions_file_type = [
    # ملفات النصوص ومستندات Office
    'txt', 'docx', 'docs', 'pdf', 'csv', 'xlsx', 'html', 'json',

    # قواعد البيانات
    'sql', 'sqlite', 'db', 'bson', 'cql', 'neo4j',

    # ملفات الإعدادات والتكوين
    'xml', 'yaml', 'yml', 'ini', 'conf', 'cfg',

    # ملفات البرمجة
    'py', 'js', 'java', 'cpp', 'c', 'rb', 'php', 'ts', 'swift', 'go',
    'cs', 'vb', 'scala', 'kt', 'rs', 'r', 'jl', 'pl', 'sh', 'bat', 'asm',
    'lua', 'dart', 'erl', 'exs', 'ml', 'clj', 'fs', 'groovy', 'ps1', 'm',
    'sas', 'sps', 'do', 'nb', 'tcl', 'ahk', 'applescript', 'vbs', 'tex',
    'md', 'org', 'hs', 'adb', 'ads', 'for', 'f', 'f90',

    # ملفات التصميم
    'css', 'scss', 'sass', 'less', 'styl'
]

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    filebook = data.get('filebook')
    model_app = data.get('model_app')
    selectedModel = model_app.get('selectedModel')
    selectedCompany = model_app.get('selectedCompany')
    chat_type = data.get('chat_type')
    chat_id = data.get('chat_id')

     # تحديد النسبة المئوية للتقليص (مثلاً 30%)
    reduction_percentage = 0.15

    # تقليص القيمة بـ 30%
    selectedModel["input_tokens"] = round(selectedModel["input_tokens"] * (1 - reduction_percentage))

    # استرجاع max_tokens بعد التقليص
    max_tokens = selectedModel.get("input_tokens")
    # يمكنك الآن استخدام max_tokens حسب الحاجة

    file_type = data.get('file_type')
    file_path = data.get('file_path')
    file_name = data.get('file_name')

    max_tokens = int(max_tokens)
    word_limit = 670

    if not question:  # التأكد من وجود السؤال
        return jsonify({"error": "Please enter a message"}), 400

    relevant_text = ""
    question_type = ""

    if chat_type == "chat":
        relevant_text = ""
        question_type = ""

        # حساب حالة الذاكرة
        total_tokens = calculate_tokens_for_memory(chat_id)  # استبدال هذه الدالة بحساب التوكينات المناسب
        memory_status = "Normal"  # الحالة الافتراضية
        if total_tokens > max_tokens:  # إذا كانت الذاكرة فوق الحد الأقصى
            memory_status = "Over Full"
        elif total_tokens > max_tokens * 0.8:  # 80% من الحد الأقصى
            memory_status = "Warning"
        elif total_tokens > max_tokens / 2:
            memory_status = "Normal - Memory is operating normally"
        else:
            memory_status = "Normal"
    else:
        # # التعامل مع الكتاب أو الملف المرفق
        # file_path = os.path.join(os.path.dirname(__file__), '..', 'public', 'storage', 'books', filebook)

        # try:
        #     with open(file_path, 'r', encoding='utf-8') as file:
        #         book_content = file.read()
        # except Exception as e:
        #     return jsonify({"error": f"حدث خطأ في فتح الملف: {e}"}), 500
        file_path__ = f"storage/books/{filebook}"  # المسار النسبي للملف
        base_url = "https://hl-ai.kulshy.online"  # الرابط الأساسي للموقع

        # استدعاء الوظيفة
        book_content = fetch_and_read_file(file_path__, base_url)
        book_content = clean_text(book_content)
        question_type = analyze_question(question)
        relevant_text = retrieve_relevant_text(question, book_content, word_limit)
        memory_status = ""


    # استدعاء دالة generate_response مع النص المستخرج
    response = {
        'answer': generate_response(relevant_text, question, question_type, selectedModel, selectedCompany, chat_id, chat_type, max_tokens, files=file_path ,file_type=file_type, file_name=file_name),
        'book_piece': relevant_text,
        'question': question,
        'memory_status': memory_status # إضافة حالة الذاكرة
    }

    return jsonify({"response": response})




def fetch_and_read_file(file_path__, base_url):
    """
    يتحقق من وجود الملف محليًا. إذا لم يكن موجودًا، يتم تحميله من الرابط.
    يقرأ محتوى الملف ويعيده.

    Args:
        file_path (str): المسار النسبي للملف (مثل "storage/books/filename.txt").
        base_url (str): الرابط الأساسي لتحميل الملفات.

    Returns:
        str: محتوى الملف إذا نجح.
        dict: رسالة خطأ إذا حدثت مشكلة.
    """
    # إعداد المسار المحلي لتخزين الملفات
    storage_dir = os.path.join(os.path.dirname(__file__), '..', 'public', 'storage', 'books')
    os.makedirs(storage_dir, exist_ok=True)  # تأكد من وجود المجلد

    # تحديد المسار المحلي الكامل
    local_file_path = os.path.join(storage_dir, os.path.basename(file_path__))

    try:
        # التحقق من وجود الملف محليًا
        if not os.path.exists(local_file_path):
            # تحميل الملف من الرابط إذا لم يكن موجودًا
            file_url = f"{base_url}/{file_path__}"
            response = requests.get(file_url, stream=True)
            response.raise_for_status()  # التحقق من وجود أي خطأ في الطلب

            # حفظ الملف محليًا
            with open(local_file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

        # قراءة محتوى الملف من النسخة المحلية
        with open(local_file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        return content

    except requests.exceptions.RequestException as e:
        return {"error": f"فشل تحميل الملف من الرابط: {e}"}
    except Exception as e:
        return {"error": f"حدث خطأ أثناء قراءة الملف: {e}"}


@app.route('/clear_memory', methods=['POST'])
def clear_memory():
    chat_id = request.json.get("chat_id")
    if chat_id in conversation_history_dict:
        conversation_history_dict[chat_id] = []
        return jsonify({"message": "Memory cleared successfully."}), 200
    return jsonify({"error": "Chat ID not found."}), 404

file = None

@app.route('/upload-file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:  # التأكد من وجود الملف
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':  # التأكد من أن هناك ملفًا
        return jsonify({'error': 'No selected file'}), 400

    # الحصول على الامتداد
    _, ext = os.path.splitext(file.filename.lower())

    if not (ext in valid_image_extensions or ext in valid_doc_extensions):
        return jsonify({'error': 'Invalid file format, please upload a PNG, JPG, TXT, DOCX, or PDF file'}), 400

    # حفظ الملفات في مجلد "uploads"
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")  # إضافة الطابع الزمني
    upload_folder = os.path.join("public", "uploads")
    os.makedirs(upload_folder, exist_ok=True)  # إنشاء المجلد إذا لم يكن موجودًا
    new_filename = f"{file.filename}_{timestamp}{ext}"  # إضافة التاريخ إلى اسم الملف
    file_path = os.path.join(upload_folder, new_filename)
    file.save(file_path)

    if ext in valid_image_extensions:
        file_type = 'image'
    elif ext in valid_doc_extensions:
        file_type = 'document'
    else:
        file_type = 'unknown'  # حالة نادرة جدا

    return jsonify({'message': f'{file_type.capitalize()} uploaded successfully', 'file_path': file_path, 'file_name':new_filename, 'file_type': ext[1:]}), 200


def generate_response(relevant_text, question, question_type, selectedModel, selectedCompany, chat_id, chat_type, max_tokens, files=None, file_type=None, file_name=None):
    prompt = f" انت خبير اكاديمي اقرأ الكتاب التالي وأجب فقط باستخدام المعلومات من الكتاب :\n\n{relevant_text}\n\nالسؤال: {question}"
    if chat_type != "chat":
        if question_type == "boolean":
            prompt += "\n\nأجب بـ 'صحيح' أو 'خطأ' بناءً على الكتاب واذكر سبب اختيار الاجابة من الكتاب ."
        elif question_type == "short_answer":
            prompt += "\n\nأجب بإجابة تستند إلى المعلومات الكتاب."
        elif question_type == "multiple_choice":
            prompt += "\n\n اختر الإجابة الصحيحة واذكر سبب اختيار الاجابة من الكتاب ."
        elif question_type == "term":
            prompt += "\n\n الجب على المصطلح الاتي"
        elif question_type == "ai":
            prompt = f"\n\n: \n\n{relevant_text}\n\nالسؤال: {question} \n\nأنت خبير أكاديمي. استخدم الذكاء الاصطناعي الخاص بك لحل السؤال."
        elif question_type == "scientific_problem":
            prompt += "\n\n أجب على المسألة العلمية التالية استنادًا إلى الكتاب المعطى."
    else:
        prompt = question

    response = ""
    """
    دالة لمعالجة المحادثة باستخدام منصة OpenAI أو Gemini بناءً على إعدادات المستخدم.
    """
    model_key = selectedModel.get("model")  # مثل "gpt-4o"
    api_key = selectedModel.get("api_key")  # المفتاح API
    model_name = selectedModel.get("name_model")  # مثل "GPT 4o"
    base_url = selectedModel.get("base_url")  # عنوان قاعدة البيانات أو الAPI
    input_tokens = selectedModel.get("input_tokens")
    output_tokens = selectedModel.get("output_tokens")
    response = ""

    if selectedCompany.get('name') == "Open Ai":
        # إعداد المفتاح API للنموذج
        openai.api_key = api_key
        openai.api_base = base_url
        # تحديد مسار الصورة (إذا كانت موجودة)
        # image_url = upload_to_openai("",  api_key=api_key)

        try:
                # إذا كانت الصورة موجودة أو لا، يتم إرسال الرسالة فقط
            if files:
                # إذا كانت الصورة موجودة، نقوم بتحديث سجل المحادثة باستخدام رابط الصورة
                updated_history, memory_error = update_conversation_history(chat_id, prompt, max_tokens, "openai", files, file_type, file_name)
            else:
                # إذا لم تكن الصورة موجودة، نرسل الرسالة فقط بدون رابط الصورة
                updated_history, memory_error = update_conversation_history(chat_id, prompt, max_tokens, "openai")

            # تحقق إذا كانت الذاكرة ممتلئة
            if memory_error:
                return memory_error  # إرجاع رسالة توضح أن الذاكرة ممتلئة

            if chat_type == "chat":
               # حساب حالة الذاكرة
                total_tokens = calculate_tokens_for_memory(chat_id)  # استبدال هذه الدالة بحساب التوكينات المناسب
                total_tokens = int(total_tokens)

                if total_tokens > max_tokens:  # إذا كانت الذاكرة فوق الحد الأقصى
                    response = "Memory is full. Please delete the memory or start a new chat." # حالة "over Full"
                    return response # حالة "over Full"
                else:
                    # استخدام واجهة API الجديدة مع سجل المحادثة
                    stream = openai.ChatCompletion.create(
                        temperature=0.2,
                        top_p=1.0,
                        model=model_key,
                        messages=updated_history,
                        max_tokens=output_tokens,
                        stream=True
                    )
            else:
                # استخدام واجهة API الجديدة بدون سجل المحادثة
                stream = openai.ChatCompletion.create(
                    model=model_key,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=output_tokens,
                    stream=True
                )

            # معالجة البيانات المستلمة من البث (stream)
            for chunk in stream:
                if chunk.get('choices') and chunk['choices'][0].get('delta') and chunk['choices'][0]['delta'].get('content'):
                    response += chunk['choices'][0]['delta']['content']  # إضافة الاستجابة المتلقاة

            if chat_type == "chat":
                # إضافة استجابة النموذج إلى سجل المحادثة
                conversation_history_dict[chat_id].append({"role": "assistant", "content": response})

        except openai.error.OpenAIError as e:
            response = f"حدث خطأ أثناء معالجة الطلب: {str(e)}"
        except Exception as e:
            response = f"حدث خطأ غير متوقع: {str(e)}"

        return response

    elif selectedCompany.get('name') == "Gemini":
        # تهيئة API مع المفتاح
        genai.configure(api_key=api_key)
        if files != None and file_type == 'png' or file_type == 'jpg' or file_type == 'jpeg':
            files = upload_to_gemini(files, mime_type="image/jpeg"),

        # تكوين إعدادات التوليد
        generation_config = {
            "temperature": 0.2,
            "top_p": 1.0,
            "top_k": 40,
            "max_output_tokens": output_tokens,
            "response_mime_type": "text/plain",
        }

        # تحديد النموذج التوليدي
        model = genai.GenerativeModel(
            model_name=model_key,
            generation_config=generation_config
        )

        if chat_type == "chat":
            # تحديث سجل المحادثة باستخدام دالة الذاكرة لمنصة Gemini
            updated_history, memory_error = update_conversation_history(chat_id, prompt, max_tokens, "gemini", files, file_type, file_name)

            # تحقق إذا كانت الذاكرة ممتلئة
            if memory_error:
                return memory_error  # إرجاع رسالة توضح أن الذاكرة ممتلئة
            # حساب حالة الذاكرة
            total_tokens = calculate_tokens_for_memory(chat_id)  # استبدال هذه الدالة بحساب التوكينات المناسب
            total_tokens = int(total_tokens)

            if total_tokens > max_tokens:  # إذا كانت الذاكرة فوق الحد الأقصى
                response = "Memory is full. Please delete the memory or start a new chat."
                return response
            else:
                # بدء جلسة المحادثة مع السجل المحدث
                chat_session = model.start_chat(history=updated_history["contents"])
        else:

            chat_session = model.start_chat(history=[{
                "role": "user",
                "parts": [prompt],
            }])

        # إرسال الرسالة في المحادثة
        try:
            # إرسال الرسالة للحصول على الاستجابة
            response_obj = chat_session.send_message(prompt)

            # إذا كانت الاستجابة تحتوي على نص، نضيفه
            if hasattr(response_obj, 'text'):
                response = response_obj.text  # استخدام الرد المستلم من النموذج

            # التحقق من وجود خاصية is_final (إذا كانت موجودة في الاستجابة)
            if hasattr(response_obj, 'is_final') and response_obj.is_final:
                return None

            # إذا كانت المحادثة من نوع 'chat'، نحدث سجل المحادثة بالرد
            if chat_type == "chat":
                updated_history, memory_error = update_conversation_history(chat_id, response, max_tokens, "gemini")
                if memory_error:
                    return memory_error  # إرجاع رسالة توضح أن الذاكرة ممتلئة

        except Exception as e:
            response = f"حدث خطأ غير متوقع: {str(e)}"

        return response
    else:

        url = base_url
        api_key = api_key

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        if chat_type == "chat":
            # إذا لم تكن الصورة موجودة، نرسل الرسالة فقط بدون رابط الصورة
            updated_history, memory_error = update_conversation_history(chat_id, prompt, max_tokens, "openai")
            # تحقق إذا كانت الذاكرة ممتلئة
            if memory_error:
                return memory_error  # إرجاع رسالة توضح أن الذاكرة ممتلئة
        else:
            updated_history = [{"role": "user", "content": prompt}]

        data = {
            "model": model_key,
            "messages": updated_history,
            "max_tokens": output_tokens,
            "temperature": 0.2,
            "top_p": 1.0,
            "top_k": 100,
            "repetition_penalty": 1,
            "stop": ["<|eot_id|>", "<|eom_id|>"],
            "stream": True
        }

        response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)

        full_response = ""
        for chunk in response.iter_lines():
            if chunk:
                line = chunk.decode('utf-8')
                if line.startswith("data: "):
                    line = line[6:]
                try:
                    chunk_data = json.loads(line)
                    # التحقق من أن "choices" تحتوي على عناصر قبل الوصول إليها
                    if "choices" in chunk_data and len(chunk_data["choices"]) > 0 and "delta" in chunk_data["choices"][0]:
                        content = chunk_data["choices"][0]["delta"].get("content", "")
                        full_response += content
                except json.JSONDecodeError as e:
                    er = f"Error decoding JSON: {e}"

        if chat_type == "chat":
            # إضافة استجابة النموذج إلى سجل المحادثة
            conversation_history_dict[chat_id].append({"role": "assistant", "content": full_response})

        return full_response

def clean_text(text):
    # تحويل النص إلى أحرف صغيرة
    text = text.lower()
    # الحفاظ على الأرقام التي تحتوي على فواصل عشرية
    text = re.sub(r'(?<=\d),(?=\d)', '،', text)  # استبدال الفواصل العشرية بفواصل عربية
    # إزالة أي رموز غير مرغوب بها باستثناء علامات الترقيم الأساسية
    text = re.sub(r'[^\w\s\.,،\؟\!]', '', text)
    return text.strip()

def analyze_question(question):
    question = clean_text(question)
    if re.search(r'\b(هل|أليس|هل تعتبر|هل يمكن|صحيح أم خطأ|هل هذا|boolean)\b', question):
        return "boolean"
    elif re.search(r'\b(ما هو|عرف|اذكر|احسب|حل|short_answer)\b', question):
        return "short_answer"
    elif re.search(r'\b(اختر|اختر الإجابة الصحيحة|اختيار من متعدد|multiple_choice)\b', question):
        return "multiple_choice"
    elif re.search(r'\b(حل|احسب|كيف|تفسير|فسر|scientific_problem)\b', question):
        return "scientific_problem"
    elif re.search(r'\b(حل|المصطلح الاتي|المصطلح الاتي|المصطلح|مصطلح|term)\b', question):
        return "term"
    elif re.search(r'\b(ai|Ai)\b', question):
        return "ai"
    else:
        return "long_answer"

def retrieve_relevant_text(question, book_content, word_limit, min_similarity=0.1):
    # التحقق من عدد الكلمات في النص بالكامل
    if len(book_content.split()) <= word_limit:
        return book_content

    # تقسيم النص إلى جمل
    sentences = re.split(r'(?<=[\.،\؟\!])\s+', book_content)
    if not sentences:
        return "النص فارغ أو لا يحتوي على جمل مفهومة."

    # حساب التشابه باستخدام TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))  # نطاق n-grams محدود للتشابه الأقل
    vectors = vectorizer.fit_transform([question] + sentences)
    cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    # ترتيب الجمل بناءً على التشابه
    sorted_indices = cosine_similarities.argsort()[::-1]
    selected_sentences = []
    total_words = 0

    for index in sorted_indices:
        sentence = sentences[index]
        similarity_score = cosine_similarities[index]

        # التحقق من درجة التشابه الدنيا
        if similarity_score < min_similarity:
            break

        sentence_word_count = len(sentence.split())

        # التوقف إذا تم الوصول إلى الحد الأقصى للكلمات
        if total_words + sentence_word_count > word_limit:
            break

        selected_sentences.append(sentence)
        total_words += sentence_word_count

    # إذا كانت النتيجة غير كافية، يتم إضافة جمل أقل تشابهًا حتى الوصول للحد المطلوب
    if total_words < word_limit:
        for index in sorted_indices[len(selected_sentences):]:
            sentence = sentences[index]
            sentence_word_count = len(sentence.split())

            if total_words + sentence_word_count > word_limit:
                break

            selected_sentences.append(sentence)
            total_words += sentence_word_count

    return ' '.join(selected_sentences) if selected_sentences else "لا توجد جمل مشابهة كافية للإجابة على السؤال."

def upload_to_gemini(path, mime_type=None):
  """Uploads the given file to Gemini.

  See https://ai.google.dev/gemini-api/docs/prompting_with_media
  """
  file = genai.upload_file(path, mime_type=mime_type)
  return file

def upload_to_openai(path, api_key=""):
    """يرتفع بالصورة إلى OpenAI باستخدام API المناسب"""
    openai.api_key = api_key

    if not os.path.exists(path):
        return None

    with open(path, "rb") as file:
        try:
            # رفع الصورة إلى OpenAI
            response = openai.Image.create(file=file, purpose='answers')

            if 'data' in response and len(response['data']) > 0:
                file_url = response['data'][0]['url']
                return file_url
            else:
                return None

        except openai.error.APIError as e:
            return None
        except Exception as e:
            return None

# إنشاء الترميز المناسب (اختياري: يمكنك اختيار الترميز المناسب حسب النموذج مثل "gpt-4")
encoding = tiktoken.get_encoding("cl100k_base")  # يمكنك تعديل الترميز حسب الحاجة

def calculate_tokens(messages):
    total_tokens = 0

    for msg in messages:
        # تحقق من وجود 'text' أو 'content' أو 'parts' في الرسالة
        if 'text' in msg:
            total_tokens += len(encoding.encode(msg['text']))
        elif 'content' in msg:
            total_tokens += len(encoding.encode(msg['content']))
        elif 'parts' in msg:
            for part in msg['parts']:
                if isinstance(part, dict) and 'text' in part:
                    total_tokens += len(encoding.encode(part['text']))
                elif isinstance(part, str):
                    # افترض أن كل ملف له نسبة معينة من التوكينات، يمكنك تعديل القيمة حسب احتياجك
                    total_tokens += 100  # أو استعمال بعض المنطق لحساب التوكينات بناءً على حجم الملف أو نوعه

    return total_tokens

conversation_history_dict = {}

def update_conversation_history(chat_id, user_message, max_tokens, platform, files=None, file_type=None, file_name=None):
    if chat_id not in conversation_history_dict:
        conversation_history_dict[chat_id] = []

    # حساب عدد التوكينات المطلوب للرسالة الجديدة
    new_message_tokens = len(user_message.split()) * 1.33
    required_tokens = new_message_tokens * 2  # مضاعفة الحد الأدنى المطلوب

    # حساب العدد الكلي للتوكينات قبل إضافة الرسالة الجديدة
    total_tokens = calculate_tokens(conversation_history_dict[chat_id])
    # حذف الرسائل القديمة حتى يتم تحرير المساحة المطلوبة
    while (total_tokens + new_message_tokens) > max_tokens:
        if len(conversation_history_dict[chat_id]) >= 2:
            # حذف رسالتين
            conversation_history_dict[chat_id].pop(0)  # حذف الرسالة الأولى
            conversation_history_dict[chat_id].pop(0)  # حذف الرسالة الثانية
        elif conversation_history_dict[chat_id]:
            # إذا كانت هناك رسالة واحدة فقط، احذفها
            conversation_history_dict[chat_id].pop(0)

        # تحديث عدد الرموز بعد الحذف
        total_tokens = calculate_tokens(conversation_history_dict[chat_id])


    # إضافة الرسالة الجديدة
    if platform == "openai":
        if files:
            if file_type in valid_image_extensions_file_type:
                conversation_history_dict[chat_id].append({"role": "user", "content": "No found image"+ user_message})
            elif file_type in valid_doc_extensions_file_type:
                # نفس الشيء هنا باستخدام "in"
                content_file = process_file(files)
                user_message += "Name File: "+file_name+" content File:" + content_file +" "+ user_message
                conversation_history_dict[chat_id].append({"role": "user", "content": user_message})

            else:
                conversation_history_dict[chat_id].append({"role": "user", "content": "No content found in file"+ user_message})
        else:
            conversation_history_dict[chat_id].append({"role": "user", "content": user_message})
    elif platform == "gemini":
        if files:
            if file_type in valid_image_extensions_file_type:
                conversation_history_dict[chat_id].append({
                    "role": "user",
                    "parts": [files[0], {"text": user_message}]
                })
            elif file_type in valid_doc_extensions_file_type:  # نفس الشيء هنا باستخدام "in"
                content_file = process_file(files)
                user_message += "Name File: "+file_name+" content File: " + content_file + user_message
                conversation_history_dict[chat_id].append({
                    "role": "user",
                    "parts": [{"text": user_message}]
                })
            else:
                conversation_history_dict[chat_id].append({
                    "role": "user",
                    "parts": [{"text":"No content found in file"+ user_message}]
                })
        else:
            conversation_history_dict[chat_id].append({
                "role": "user",
                "parts": [{"text": user_message}]
            })

    # طباعة الحالة النهائية
    total_tokens = calculate_tokens(conversation_history_dict[chat_id])

    # معالجة الرسائل للمنصات المختلفة
    if platform == "openai":
        return conversation_history_dict[chat_id], None
    elif platform == "gemini":
        gemini_history = []
        for msg in conversation_history_dict[chat_id]:
            if 'content' in msg:
                gemini_history.append({
                    "role": msg['role'],
                    "parts": [{"text": msg["content"]}]
                })
            elif 'parts' in msg:
                gemini_history.append({
                    "role": msg['role'],
                    "parts": msg["parts"]
                })
        return {"contents": gemini_history}, None
    else:
        return None, "منصة غير مدعومة."

# دالة لحساب التوكينات للذاكرة
def calculate_tokens_for_memory(chat_id):
    total_tokens = 0
    if chat_id in conversation_history_dict:
        total_tokens = calculate_tokens(conversation_history_dict[chat_id])  # أو أي منصة أخرى
    return total_tokens


def process_file(file_path):
    """تحديد نوع الملف واستخراج النص بناءً على نوعه"""
    _, ext = os.path.splitext(file_path.lower())

    # التحقق من نوع الملف بناءً على الامتداد
    if ext == '.txt':
        return extract_text_from_txt(file_path)
    elif ext in ['.docx', '.docs']:
        return extract_text_from_docx(file_path)
    elif ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.csv':
        return extract_text_from_csv(file_path)
    elif ext == '.xlsx':
        return extract_text_from_excel(file_path)
    elif ext == '.html':
        return extract_text_from_html(file_path)
    elif ext == '.json':
        return extract_text_from_json(file_path)
    elif ext in ['.sql', '.sqlite', '.db', '.bson', '.cql', '.neo4j']:
        return extract_text_from_database(file_path)
    elif ext in ['.xml', '.yaml', '.yml', '.ini', '.conf', '.cfg']:
        return extract_text_from_config(file_path)
    elif ext in [
        '.py', '.js', '.java', '.cpp', '.c', '.rb', '.php', '.ts', '.swift', '.go',
        '.cs', '.vb', '.scala', '.kt', '.rs', '.r', '.jl', '.pl', '.sh', '.bat', '.asm',
        '.lua', '.dart', '.erl', '.exs', '.ml', '.clj', '.fs', '.groovy', '.ps1', '.m',
        '.sas', '.sps', '.do', '.nb', '.tcl', '.ahk', '.applescript', '.vbs', '.tex',
        '.md', '.org', '.hs', '.adb', '.ads', '.for', '.f', '.f90', '.css', '.scss', '.sass',
        '.less', '.styl'
    ]:
        return extract_text_from_code(file_path)
    else:
        return f"Unsupported file type: {ext}"

def extract_text_from_txt(file_path):
    """استخراج النص من ملف نصي (.txt)"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except Exception as e:
        return f"Error reading .txt file: {str(e)}"

def extract_text_from_docx(file_path):
    """استخراج النص من ملف Word (.docx)"""
    try:
        doc = Document(file_path)
        text = '\n'.join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        return f"Error reading .docx file: {str(e)}"

def extract_text_from_pdf(file_path):
    """استخراج النص من ملف PDF"""
    try:
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + '\n'
        return text
    except Exception as e:
        return f"Error reading .pdf file: {str(e)}"

def extract_text_from_csv(file_path):
    """استخراج النص من ملف CSV"""
    try:
        text = ""
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                text += ', '.join(row) + '\n'
        return text
    except Exception as e:
        return f"Error reading .csv file: {str(e)}"

def extract_text_from_excel(file_path):
    """استخراج النص من ملف Excel (.xlsx)"""
    try:
        df = pd.read_excel(file_path)
        text = df.to_string(index=False)
        return text
    except Exception as e:
        return f"Error reading .xlsx file: {str(e)}"

def extract_text_from_html(file_path):
    """استخراج النص من ملف HTML"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
            text = soup.get_text(separator='\n')
        return text
    except Exception as e:
        return f"Error reading .html file: {str(e)}"

def extract_text_from_json(file_path):
    """استخراج النص من ملف JSON"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        text = json.dumps(data, indent=4, ensure_ascii=False)
        return text
    except Exception as e:
        return f"Error reading .json file: {str(e)}"

def extract_text_from_database(file_path):
    """استخراج النص من ملف قاعدة بيانات أو SQL"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except Exception as e:
        return f"Error reading database file: {str(e)}"

def extract_text_from_config(file_path):
    """استخراج النص من ملفات الإعدادات (XML, YAML, INI)"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except Exception as e:
        return f"Error reading config file: {str(e)}"

def extract_text_from_code(file_path):
    """استخراج النص من ملفات البرمجة"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except Exception as e:
        return f"Error reading code file: {str(e)}"




if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5002)



