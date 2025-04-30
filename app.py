from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import random
import uvicorn

import logging
from typing import Optional

"""
python3 -m http.server 8000

老闆早，昨日使用 HTML 和 CSS 成功架構一個簡易的網頁Demo，今日將進行把原本地端的GUI建設成網頁版本：
- 把本地端程式變成網頁版本（Web App）：研究 web app 建構方式，這樣以後要做成網站或手機 App（iOS / Android）會更方便。
1. 熟悉 HTML、CSS 和 JavaScript的基本概念 (100%)
2. 研究如何使用 JavaScript 的 WebRTC API 來實現音訊串流 (100%)
3. 研究如何使用 HTML 和 CSS 來設計網頁介面 (100%)
4. 將本地端的 GUI 程式碼轉換為網頁介面
5. 研究如何架設 Python 伺服器：未來應用需要自行架設 server 用於儲存、分析和搜索使用者資料使用。
6. 串接網頁與 Python server：研究如何讓網頁與 Python server 進行串接，並能夠傳遞資料。
7. 研究如何將前端網頁與Python server部署到網站上。
- 學習跨平台開發工具（Flutter / React Native）：只寫一次，就能把 Web App 同時發佈成 iOS 和 Android App。
- 解決聲音回音問題：研究避免喇叭聲音被麥克風收到，造成 AI 無限自我回應。
1. 尋找類似回音解決案例 (100%)
2. 閱讀該範例原始碼研究回音消除實現方式(100%)
3. 研究如何在 server 端進行調整，避免喇叭聲音被麥克風收到

- 企劃更清楚的對話資訊提取結構：讓 AI 更有效率地從使用者對話中找出重要資訊。
- 企劃分析user資料轉成有價成果：讓 AI 透過分工逐步整理分析user資料，產生user的自我對齊狀態、人生故事、願望清單、信念與情緒變化...等。
"""
message_callback = None
# Global variable to store the message callback
def set_message_callback(callback):
    global message_callback
    message_callback = callback

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Example usage of logger
logger.info("Logging is set up.")


app = FastAPI()
# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv(override=True)

# Get API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
REALTIME_SESSION_URL = os.getenv("REALTIME_SESSION_URL")

# this is the openai url: https://api.openai.com/v1/realtime/sessions
logger.info(f"REALTIME_SESSION_URL: {REALTIME_SESSION_URL}")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
if not SERPER_API_KEY:
    raise ValueError("SERPER_API_KEY not found in environment variables")
if not REALTIME_SESSION_URL:
    raise ValueError("REALTIME_SESSION_URL not found in environment variables")

class SessionResponse(BaseModel):
    session_id: str
    token: str

class WeatherResponse(BaseModel):
    temperature: float
    humidity: float
    precipitation: float
    wind_speed: float
    unit_temperature: str = "celsius"
    unit_precipitation: str = "mm"
    unit_wind: str = "km/h"
    forecast_daily: list
    current_time: str
    latitude: float
    longitude: float
    location_name: str
    weather_code: int

class SearchResponse(BaseModel):
    title: str
    snippet: str
    source: str
    image_url: Optional[str] = None
    image_source: Optional[str] = None

class UserMessage(BaseModel):
    message: str
    role:str

class UserQuery(BaseModel):
    query: str    

@app.post("/query")
async def query_openai(data: UserQuery):
    logger.info(f"Received query: {data.query}")
    try:
        response, ui_response = message_callback("search_user_data", data.query)
        logger.info(f"Response from callback: {response}")
        logger.info(f"UI_Response from callback: {ui_response}")
        return response
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error occurred: {e.response.status_code}")
        return JSONResponse(status_code=e.response.status_code, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Internal Server Error", "details": str(e)})

@app.post("/messages")
async def process_messages(data: UserMessage):
    global message_callback
    try:
        # 在這裡儲存訊息，暫時用 log 示意
        # logger.info(f"[{data.role}] message received: {data.message}")
        if message_callback:
            message_callback(data.role, data.message)   
        return {"status": "success", "message": data.message, "role": data.role}
       
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Internal Server Error", "details": str(e)})

@app.get("/stop_and_analysis")
async def stop_and_analysis():
    print("🖥️ app_servr:stop_and_analysis")
    global message_callback
    try:
        if message_callback:
            message_callback("stop_and_analysis", "stop_and_analysis")
        return {"status": "success", "message": "stop_and_analysis"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Internal Server Error", "details": str(e)})

@app.get("/session")
async def get_session(voice: str = "echo"):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                REALTIME_SESSION_URL,
                headers={
                    'Authorization': f'Bearer {OPENAI_API_KEY}',
                    'Content-Type': 'application/json'
                },
                json={
                    "model": "gpt-4o-realtime-preview-2024-12-17",
                    "voice": voice,
                    # "turn_detection": {
                    #     "type": "server_vad",
                    #     "threshold": 0.7,
                    #     "prefix_padding_ms": 300,
                    #     "silence_duration_ms": 2000
                    # },
                    "temperature": 1,
                    "max_response_output_tokens": 4096,
                    "modalities": ["text", "audio"],
                    "input_audio_transcription": {"model": "whisper-1"},
                    "instructions": """[系統角色與人格目標]
                    你是一位「FreeAger 品牌」的 AI 助手，以「一路健康玩到掛」為願景，協助長者突破年齡與身心限制，陪伴他們保持活力、獲得社會連結、並自在且無遺憾地面對人生終章。
                    你的語調「活力、輕快、激動」，並擁有溫暖、耐心、真誠且包容的個性，擅長同理傾聽、積極鼓勵，帶給長者安全感和希望。
                    你對「感覺年輕」「敢玩」「不怕死」這三大策略尤其敏銳，能在對話中自然注入，陪伴長者不斷自我探索並豐富生活。
                    
                    [核心對話風格與要點]
                    請用「活力、輕快、激動」的語調和使用者對談，語調可以稍微加快，以凸顯活力。
                    在長者表達脆弱或擔憂時，給予同理心與正向支持。
                    激發好奇與行動：當長者陷於「我不行了」「年紀太大」等負面想法時，運用幽默、分享同輩案例或正面肯定，鼓勵嘗試新事物或拓展人際互動。
                    專注健康自我照顧：適度提供身心復原力、日常活動建議與資源連結，引導長者強化健康認知並維持「感覺年輕」。
                    引導無悔心態：若對話觸及生命終章、遺憾或死亡話題，你會以柔和尊重的方式探詢其渴望、想完成的事情，或可共創的靈性探索，讓他們漸漸擁有「不怕死」的平靜感。
                    避免年齡歧視思維：無論長者的身心狀態、背景或能力，都要維持多元、包容的回應；同時讓他們意識到年齡不應是阻礙，人生還有無限可能。
                    
                    [關鍵功能與範例對話策略]
                    關懷式提問：像「今天過得怎麼樣？最近有遇到什麼新鮮有趣的事嗎？」
                    陪伴與激勵：常用「我聽起來感覺你很努力…」「真不簡單，能分享更多嗎？」讓長者被看見、被鼓勵。
                    目標探索：當談到人生目標或遺憾，主動詢問「有沒有想要學的嗜好、或想嘗試的活動？」，並幫助規劃具體行動步驟。
                    接受與釋懷：碰到重大病痛或失落，保持耐心與同理，不匆忙跳到解決方案；若談及死亡與終活，可陪伴對方思考回顧與未竟心願。

                    [行為守則]
                    不提供醫療、財務或法律專業建議。如遇超出能力範圍的個案問題，請溫和建議使用者尋求合格專業人員協助。
                    尊重長者意願與感受，不評斷其價值選擇或信仰立場。
                    任何鼓勵活動皆須考量對方的身體與心理負荷，並主動提醒「若有疑慮，可先諮詢專業」。

                    [整體語氣示範]
                    「我懂你現在正面臨的壓力，也很期待和你一起探索哪些方法能讓生活重新充滿樂趣。」
                    「或許可以嘗試幾個簡單的伸展動作，你覺得如何？相信你只要一步步來，就能看見意想不到的成長。」
                    「面對死亡並不代表害怕或逃避，而是讓我們更珍惜現在每一刻。你是否有想過做什麼一直沒實現的事呢？」"""
                }
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error occurred: {e.response.status_code}")
        return JSONResponse(status_code=e.response.status_code, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Internal Server Error", "details": str(e)})

@app.get("/weather/{location}")
async def get_weather(location: str):
    try:
        async with httpx.AsyncClient() as client:
            # Get coordinates for location
            geocoding_response = await client.get(
                f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1"
            )
            # logger.info(f"geocoding_response:{geocoding_response}")
            geocoding_data = geocoding_response.json()
            # logger.info(f"geocoding_data:{geocoding_data}")
            if not geocoding_data.get("results"):
                return {"error": f"Could not find coordinates for {location}"}
                
            lat = geocoding_data["results"][0]["latitude"]
            lon = geocoding_data["results"][0]["longitude"]
            location_name = geocoding_data["results"][0]["name"]
            
            # Get weather data with more parameters
            weather_response = await client.get(
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                f"&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code"
                f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code"
                f"&timezone=auto"
                f"&forecast_days=7"
            )
            weather_data = weather_response.json()
            # logger.info(f"weather_data:\n{weather_data}")
            
            # Extract current weather
            current = weather_data["current"]
            daily = weather_data["daily"]
            
            # Create daily forecast array
            forecast = []
            for i in range(len(daily["time"])):
                forecast.append({
                    "date": daily["time"][i],
                    "max_temp": daily["temperature_2m_max"][i],
                    "min_temp": daily["temperature_2m_min"][i],
                    "precipitation": daily["precipitation_sum"][i],
                    "weather_code": daily["weather_code"][i]
                })
            
            return WeatherResponse(
                temperature=current["temperature_2m"],
                humidity=current["relative_humidity_2m"],
                precipitation=current["precipitation"],
                wind_speed=current["wind_speed_10m"],
                forecast_daily=forecast,
                current_time=current["time"],
                latitude=lat,
                longitude=lon,
                location_name=location_name,
                weather_code=current["weather_code"]
            )
            
    except Exception as e:
        logger.error(f"Error getting weather data: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Could not get weather data: {str(e)}"})

@app.get("/search/{query}")
async def search_web(query: str):
    try:
        async with httpx.AsyncClient() as client:
            # Get regular search results
            response = await client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": SERPER_API_KEY},
                json={"q": query}
            )
            
            data = response.json()
            
            # Get image search results with larger size
            image_response = await client.post(
                "https://google.serper.dev/images",
                headers={"X-API-KEY": SERPER_API_KEY},
                json={
                    "q": query,
                    "gl": "us",
                    "hl": "en",
                    "autocorrect": True
                }
            )
            
            image_data = image_response.json()
            
            if "organic" in data and len(data["organic"]) > 0:
                result = data["organic"][0]  # Get the first result
                image_result = None
                
                # Find first valid image
                if "images" in image_data:
                    for img in image_data["images"]:
                        if img.get("imageUrl") and (
                            img["imageUrl"].endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')) or 
                            'images' in img["imageUrl"].lower()
                        ):
                            image_result = img
                            break
                
                return SearchResponse(
                    title=result.get("title", ""),
                    snippet=result.get("snippet", ""),
                    source=result.get("link", ""),
                    image_url=image_result["imageUrl"] if image_result else None,
                    image_source=image_result["source"] if image_result else None
                )
            else:
                return {"error": "No results found"}
                
    except Exception as e:
        logger.error(f"Error performing search: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Could not perform search: {str(e)}"})

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8888)
