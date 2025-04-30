# -*- coding: utf-8 -*-
import tool_ai_chatbot as ai_chatbot
import ai_pet_prompt as prompt
import app as server_app
import faiss
import time
from datetime import datetime
import threading
import xml.etree.ElementTree as ET
import os
from dotenv import load_dotenv
import pandas as pd
from collections import defaultdict
import sqlite3
import json
from openai import OpenAI
import numpy as np





# Server 相關========================================================================================================
def server_callback(role, text, data=None):
    # print(f"====\server_callback:\nrole:{role}\ntext:{text}\ndata:{data}\n====")
    """確保第一條一定是使用者，且之後使用者與 AI 交替"""
    if not text.strip():
        return  # 避免存入空訊息
    
    # 觸發分析條件
    if role == "search_user_data":
        print(f"server_callback:search_user_data\nquery:{text}")
        results, ui_result = search_similar_statements_with_faiss(text, category_filter=None, subcategory_filter=None)
        return results, ui_result
    
    if role == "stop_and_analysis" and len(realtime_temp_message) > 1:
        try:
            analyze_beliefs_persona_emotions_story()
        except Exception as e:
            print(f"❌ 分析失敗：{e}")
    
    # 1. 確保第一條訊息是 "user"
    if not realtime_temp_message and role != "user":
        print("⚠️ 第一條訊息必須來自使用者，忽略此訊息")
        return

    # 2. 確保 user 和 ai 交替
    if realtime_temp_message:
        last_role = realtime_temp_message[-1]["role"]
        if last_role == role:
            realtime_temp_message[-1]["content"] += text
            print(f"⚠️ 將 {role} 上下訊息整合：{realtime_temp_message[-1]['content']}")
            return  # 跳過不符合交替規則的訊息

    message = {"role": role, "content": text}
    if message["role"] == "user" or message["role"] == "assistant":
        realtime_temp_message.append(message)
        realtime_all_message.append(message)
        

    # print(f"realtime_temp_message:\n{realtime_temp_message}")
    # print(f"realtime_all_message:\n{realtime_all_message}")
    


    if len(realtime_temp_message) > message_analysis_threshold:
        try:
            analyze_beliefs_persona_emotions_story()
        except Exception as e:
            print(f"❌ 分析失敗：{e}")
           

    


# message analysis相關========================================================================================================
def process_conversation_short_term_analysis(query="", assistant_params=None):
    global realtime_temp_message
    global start_qtgui
    ### 將暫時訊息存入永久訊息中，同時為了避免最後一條訊息爲user造成chatbot錯誤使用以下判斷
    last_user_message = None
    if realtime_temp_message and realtime_temp_message[-1]['role'] == 'user':
        last_user_message = realtime_temp_message[-1]
        realtime_temp_message.pop()
    # 更新聊天機器人的訊息
    chatbot_short_term_analysis.read_previous_messages(messages_history=realtime_temp_message)
    # 重置 `realtime_temp_message`，保留最後一筆使用者訊息（如果有）
    realtime_temp_message = [last_user_message] if last_user_message else []
    response = chatbot_short_term_analysis.assistant_with_search_tool(**assistant_params)
    chatbot_short_term_analysis.save_message_to_csv()
    chatbot_short_term_analysis.clear_all_messages()
    response_xml = ai_chatbot.extract_tagged_content_from_str(response,
                                                              f"answer_{query}")

    response_dic, response_df = parse_psychological_analysis(response_xml)
    print(f"⚠️===response:\n{response_xml}")
    print(f"response_dic:\n{response_dic}")
    print(f"response_df:\n{response_df}")
    # 將分析結果總結至 message_analysis_text
    summarize_response_dic(response_dic)
    print("message_analysis_text:\n" + "\n\n".join(message_analysis_text))

    # 將資料儲存
    save_to_db_safely(response_df)

    # return response


# 後來沒用到 long term
def process_conversation_long_term_analysis(short_term_need_summary):
    def convert_input_data_to_prompt_text(input_data: list[dict]) -> str:
        return "\n".join(
            [f"分類：{d['Subcategory']}\n敘述：{d['Statement']}\n佐證：{d['Evidence']}" for d in input_data]
        )
    short_term_need_summary_str = convert_input_data_to_prompt_text(short_term_need_summary)
    # print(f"short_term_need_summary_str:\n{short_term_need_summary_str}")
    query, assistant_params = prompt.prompt_summarize_structured_data_to_text(short_term_need_summary_str)
    response = chatbot_long_term_analysis.assistant_with_search_tool(**assistant_params)
    chatbot_long_term_analysis.save_message_to_csv()
    chatbot_long_term_analysis.clear_all_messages()
    response = ai_chatbot.extract_tagged_content_from_str(response,
                                                              f"answer_{query}")


def analyze_beliefs_persona_emotions_story():
    # 取得 query 和 assistant_params
    query1, assistant_params1 = prompt.prompt_message_analysis_beliefs()
    query2, assistant_params2 = prompt.prompt_message_analysis_persona()
    query3, assistant_params3 = prompt.prompt_message_analysis_emotions()
    query4, assistant_params4 = prompt.prompt_message_analysis_life_story()

    # 建立 thread
    threads = [
        threading.Thread(target=process_conversation_short_term_analysis, args=(query1, assistant_params1), name="analysis_beliefs"),
        threading.Thread(target=process_conversation_short_term_analysis, args=(query2, assistant_params2), name="analysis_persona"),
        threading.Thread(target=process_conversation_short_term_analysis, args=(query3, assistant_params3), name="analysis_emotions"),
        threading.Thread(target=process_conversation_short_term_analysis, args=(query4, assistant_params4), name="analysis_life_story"),
    ]

    for t in threads:
        t.start()


def analyze_and_save():
    # 執行剩餘訊息分析並儲存所有訊息
    analyze_beliefs_persona_emotions_story()
    print(f"🧠 全部訊息記錄：{realtime_all_message}")
    current_time_str = datetime.now().strftime("%Y%m%d-%H%M")
    ai_chatbot.save_message_to_csv(realtime_all_message, f"./message_record/realtime_{current_time_str}.csv")


# database相關========================================================================================================
def parse_psychological_analysis(xml_input):
    """
    解析心理分析的 XML（字串或 list），回傳：
    1. results：List[Dict]，每條語句與合併後的 evidences
    2. df：pandas DataFrame，Evidence 會以 "||" 連接
    """
    if isinstance(xml_input, list):
        xml_input = xml_input[0]

    xml_string = f"<root>{xml_input}</root>"
    tree = ET.fromstring(xml_string)

    evidence_map = defaultdict(list)
    timestamp = datetime.now().strftime("%Y/%m/%d-%H:%M")  # 自動取得目前時間

    def extract_items(section, category, subcategory):
        if section.attrib.get("has_content") == "true":
            for item in section.findall("item"):
                statement = item.findtext("statement", default="").strip()
                evidences = [e.text.strip() for e in item.findall("evidence") if e.text]
                evidence_map[(category, subcategory, statement)].extend(evidences)

    # 主迴圈
    for category_node in tree:
        category = category_node.tag
        for subcategory_node in category_node:
            subcategory = subcategory_node.tag
            extract_items(subcategory_node, category, subcategory)

    # 轉換成 list of dicts
    results = [
        {
            "Timestamp": timestamp,  # 自動加入目前時間
            "Category": key[0],
            "Subcategory": key[1],
            "Statement": key[2],
            "Evidence": " || ".join(value),  # 用 "||" 連接多個 evidence
        }
        for key, value in evidence_map.items()
    ]

    # 轉為 DataFrame
    df = pd.DataFrame(results)
    return results, df


# 將解析好的對話資料儲存至短期sqlite
def append_analysis_to_sqlite(df, db_path="table_name.db", table_name="table_name"):
    """
    將心理分析結果寫入 SQLite 資料庫，自動管理 analysis_id。
    若資料表不存在會自動建立。
    """
    print(f"=>append_analysis_to_sqlite:{df}")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 建立資料表（如果不存在）
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
            Timestamp TEXT,
            Category TEXT,
            Subcategory TEXT,
            Statement TEXT,
            Evidence TEXT,
            statement_vector TEXT,
            longterm_summary BOOLEAN DEFAULT 0
        )
    """)
    conn.commit()
    try:
        if not df.empty:
            # 儲存資料（不含 analysis_id，會自動編號）
            df[["Timestamp", "Category", "Subcategory", "Statement", "Evidence"]].to_sql(
                name=table_name,
                con=conn,
                if_exists='append',
                index=False
            )

            # 顯示目前筆數
            count = cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"✅ 已新增 {len(df)} 筆資料至 {db_path}（總計 {count} 筆）")
    except sqlite3.Error as e:
        print(f"❌ 儲存資料失敗：{e}")        
    conn.close()

# 將短期sqlite的statement+evidence進行向量化
def batch_embed_statements_from_sqlite(
        db_path="./message_record/short_term/analysis.db",
        table_name="analysis_result_db",
        batch_size=50
):
    print(f"=>batch_embed_statements_from_sqlite")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 取得尚未向量化的資料
    query = f"""
        SELECT analysis_id, Statement, Evidence
        FROM {table_name}
        WHERE statement_vector IS NULL OR statement_vector = ''
    """
    df = pd.read_sql_query(query, conn)

    if df.empty:
        print("✅ 所有資料都已向量化，無需處理。")
        return

    print(f"📦 共需向量化 {len(df)} 筆資料。每批次 {batch_size} 筆")

    # 建立新欄位合併語意文字
    df["text_for_embedding"] = df["Statement"].astype(str) + " || " + df["Evidence"].astype(str)

    # 批次處理
    for start in range(0, len(df), batch_size):
        end = start + batch_size
        batch_df = df.iloc[start:end]
        texts = batch_df["text_for_embedding"].tolist()
        ids = batch_df["analysis_id"].tolist()

        print(f"🚀 正在處理 batch {start} - {end - 1}")

        try:
            response = clientGPT.embeddings.create(model=embedding_model, input=texts)
            embeddings = [json.dumps(e.embedding) for e in response.data]

            # 寫入回 SQLite
            for analysis_id, vector in zip(ids, embeddings):
                cursor.execute(f"""
                    UPDATE {table_name}
                    SET statement_vector = ?
                    WHERE analysis_id = ?
                """, (vector, analysis_id))

            conn.commit()
            print(f"✅ 成功更新 {len(ids)} 筆資料")

        except Exception as e:
            print(f"❌ 失敗於 batch {start}-{end - 1}，錯誤：{e}")
            continue

        time.sleep(0.5)  # 安全等待，避免 hitting rate limit

    conn.close()
    print("🎉 所有向量儲存完成")


# 建立與更新faiss的向量資料庫
def build_or_update_faiss_index_from_sqlite(
        db_path="./message_record/short_term/analysis.db",
        table_name="analysis_result_db",
        index_path="./message_record/short_term/analysis_faiss.index",
        id_map_path="./message_record/short_term/analysis_id_map.json"
):
    """
    根據 SQLite 的資料，建立或更新 FAISS 向量索引。
    - 若尚未存在索引，會建立新的。
    - 若已存在索引，只會加入尚未存入的分析項目。
    """
    print("=>build_or_update_faiss_index_from_sqlite")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    # 連線資料庫
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 取得所有向量資料
    cursor.execute(f"""
        SELECT analysis_id, statement_vector
        FROM {table_name}
        WHERE statement_vector IS NOT NULL AND statement_vector != ''
    """)
    all_rows = cursor.fetchall()
    conn.close()

    if not all_rows:
        print("⚠️ 沒有任何可用向量資料，請先執行向量化流程。")
        return

    # 解析資料庫中的向量
    all_id_vector_map = {row[0]: json.loads(row[1]) for row in all_rows}

    # 檢查是否已有 index 與 id_map
    if os.path.exists(index_path) and os.path.exists(id_map_path):
        print("🔍 檢測到已有索引，進行增量更新...")

        # 載入 index 和 ID map
        index = faiss.read_index(index_path)
        with open(id_map_path, "r", encoding="utf-8") as f:
            existing_ids = json.load(f)

        existing_ids_set = set(existing_ids)

        # 篩選出未加入的向量
        new_items = [(i, v) for i, v in all_id_vector_map.items() if i not in existing_ids_set]
        if not new_items:
            print("✅ 索引已為最新，無需更新。")
            return

        new_ids, new_vectors = zip(*new_items)
        new_vectors = np.array(new_vectors).astype("float32")
        faiss.normalize_L2(new_vectors)

        index.add(new_vectors)
        faiss.write_index(index, index_path)

        updated_ids = existing_ids + list(new_ids)
        with open(id_map_path, "w", encoding="utf-8") as f:
            json.dump(updated_ids, f)

        print(f"✅ FAISS 索引已成功新增 {len(new_ids)} 筆向量。")

    else:
        print("🚀 建立全新索引...")
        all_ids = list(all_id_vector_map.keys())
        all_vectors = np.array(list(all_id_vector_map.values())).astype("float32")
        faiss.normalize_L2(all_vectors)

        index = faiss.IndexFlatIP(all_vectors.shape[1])
        index.add(all_vectors)
        faiss.write_index(index, index_path)

        with open(id_map_path, "w", encoding="utf-8") as f:
            json.dump(all_ids, f)

        print(f"✅ 已建立新的 FAISS 向量索引，共儲存 {len(all_ids)} 筆資料。")


# 使用faiss進行語意搜索
def search_similar_statements_with_faiss(
        query: str,
        db_path="./message_record/short_term/analysis.db",
        table_name="analysis_result_db",
        index_path="./message_record/short_term/analysis_faiss.index",
        id_map_path="./message_record/short_term/analysis_id_map.json",
        top_k=5,
        category_filter: str = None,
        subcategory_filter: str = None,
):
    """
    使用者 query → FAISS 查詢 → 回傳 SQLite 中的語句資料，支援類別過濾。
    """
    print("開始查詢向量資料庫")

    # 1. 將 query 轉為向量
    response = clientGPT.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_vector = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(query_vector)

    # 2. 載入 FAISS 索引與 ID 對應
    if not os.path.exists(index_path) or not os.path.exists(id_map_path):
        raise FileNotFoundError("❌ 找不到索引檔案，請先執行 build_faiss_index_from_sqlite()")

    index = faiss.read_index(index_path)
    with open(id_map_path, "r", encoding="utf-8") as f:
        id_list = json.load(f)

    # 3. 查詢最相似的向量
    scores, indices = index.search(query_vector, top_k)  # 多撈一點再篩選
    matched_ids = [id_list[i] for i in indices[0]]
    matched_scores = scores[0]

    # 4. 查詢 SQLite，加入過濾條件
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    placeholder = ",".join(["?"] * len(matched_ids))
    sql = f"""
        SELECT analysis_id, Category, Subcategory, Statement, Evidence
        FROM {table_name}
        WHERE analysis_id IN ({placeholder})
    """

    params = matched_ids

    if category_filter:
        sql += " AND Category = ?"
        params.append(category_filter)
    if subcategory_filter:
        sql += " AND Subcategory = ?"
        params.append(subcategory_filter)

    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()

    # 5. 回傳原始順序資料
    result_map = {r[0]: r for r in rows}
    results = []

    for i, match_id in enumerate(matched_ids):
        if match_id in result_map:
            analysis_id, category, subcategory, statement, evidence = result_map[match_id]
            results.append({
                "analysis_id": analysis_id,
                "category": category,
                "subcategory": subcategory,
                "statement": statement,
                "evidence": evidence,
                "score": float(matched_scores[i])
            })
    ui_result_content = f"💡 搜尋問題：{query}\n=========================\n"
    for r in results:
        ui_result_content += f"\n📖 陳述： {r['statement']}"
        ui_result_content += f"\n🧾 證據： {r['evidence']}"
        ui_result_content += f"\n🔹 相關： {r['score']:.3f}"
        ui_result_content += f"\n📌 類別： {r['category']} / {r['subcategory']}"
        ui_result_content += f"\n--------\n"
        print(f"🔹 相似度: {r['score']:.3f}")
        print(f"📌 類別: {r['category']} / {r['subcategory']}")
        print(f"📖 Statement: {r['statement']}")
        print(f"🧾 Evidence: {r['evidence']}")
        print("--------")
    return results, ui_result_content


def save_to_db_safely(response_df):
    print("save_to_db_safely")
    with db_lock:
        append_analysis_to_sqlite(response_df, db_path=short_term_db_path,
                                    table_name=short_term_table_name)
        batch_embed_statements_from_sqlite()
        build_or_update_faiss_index_from_sqlite()


# 其他========================================================================================================
def summarize_response_dic(response_dic):
    global message_analysis_text

    # 建立結構：{Category: {Subcategory: [(Statement, [Evidence])...]}}
    grouped = defaultdict(lambda: defaultdict(list))
    for entry in response_dic:
        category = entry["Category"]
        subcat = entry["Subcategory"]
        statement = entry["Statement"]
        evidence = [e.strip() for e in entry["Evidence"].split("||")]
        grouped[category][subcat].append((statement, evidence))

    # 美化輸出
    output_lines = []
    for category, subcats in grouped.items():
        output_lines.append("=" * 25)
        output_lines.append(f"[ {category} ]")
        output_lines.append("=" * 25)

        for subcat, items in subcats.items():
            output_lines.append(f"{subcat} : ")
            for statement, evidences in items:
                output_lines.append(f"  - {statement}")
                for ev in evidences:
                    output_lines.append(f"    ・{ev}")
            output_lines.append("")  # 空行分段

    result = "\n".join(output_lines).strip()
    message_analysis_text.append(result)
    return result


# 設置global變數=========================================================================================================
load_dotenv() # 設定環境變數
clientGPT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedding_model = "text-embedding-3-small"  # ✅ 嵌入模型
# 訊息管理變數
realtime_temp_message = []
realtime_all_message = []
message_analysis_threshold = 40  # 累積多少訊息進行分析
message_analysis_text = []  # 顯示訊息分析的ui文字
db_lock = threading.Lock()  # 用於防止多執行緒的同時資料庫寫入


# 創立chatbot
chatbot_short_term_analysis = ai_chatbot.AIChatbot(bot_name="bot_short_term", ai_model="claude-3-7-sonnet-20250219") # gpt-4o / claude-3-5-sonnet-20241022
chatbot_short_term_analysis.print_prompt = True
chatbot_long_term_analysis = ai_chatbot.AIChatbot(bot_name="bot_long_term", ai_model="claude-3-7-sonnet-20250219") # gpt-4o / claude-3-5-sonnet-20241022
chatbot_long_term_analysis.print_prompt = True

# db 設定
short_term_db_path = "./message_record/short_term/analysis.db"
short_term_table_name = "analysis_result_db"


# 主程式開始==============================================================================================================
if __name__ == '__main__':
    print("main.start")
    ### 開啟後端伺服器
    app_start = True
    if app_start:
        # 啟動後端伺服器
        server_app.set_message_callback(server_callback)
        server_app.uvicorn.run(server_app.app, host="0.0.0.0", port=8888)
    
        
   






