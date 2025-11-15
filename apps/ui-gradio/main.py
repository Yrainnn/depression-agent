from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import sys
import gradio as gr

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 导入你的Gradio应用
from app import build_ui

# 创建FastAPI应用
app = FastAPI(title="智能心境健康评估系统")

# 挂载静态文件（如果需要）
# app.mount("/static", StaticFiles(directory="static"), name="static")

# 设置模板目录
templates = Jinja2Templates(directory="templates")

# 存储登录界面HTML内容
LOGIN_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>智能心境评估系统</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://cdn.jsdelivr.net/npm/font-awesome@4.7.0/css/font-awesome.min.css" rel="stylesheet">
  
  <!-- 配置Tailwind自定义颜色和字体 -->
  <script>
    tailwind.config = {
      theme: {
        extend: {
          colors: {
            primary: '#4ade80', // 主色调：治愈绿
            secondary: '#a7f3d0', // 辅助色：淡绿
            accent: '#059669', // 强调色：深绿
            light: '#f0fdf4', // 浅色背景
          },
          fontFamily: {
            sans: ['Inter', 'system-ui', 'sans-serif'],
          },
        },
      }
    }
  </script>
  
  <style type="text/tailwindcss">
    @layer utilities {
      .content-auto {
        content-visibility: auto;
      }
      .bg-gradient-custom {
        background: linear-gradient(135deg, #4ade80 0%, #a7f3d0 100%);
      }
      .text-shadow {
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
      }
      .animate-float {
        animation: float 6s ease-in-out infinite;
      }
      .animate-float-delay {
        animation: float 6s ease-in-out 2s infinite;
      }
      .animate-float-slow {
        animation: float 8s ease-in-out infinite;
      }
      .fade-in {
        animation: fadeIn 0.8s ease forwards;
      }
      .slide-up {
        animation: slideUp 0.6s ease forwards;
      }
      .scale-in {
        animation: scaleIn 0.5s ease forwards;
      }
      .bg-transition {
        transition: opacity 1.5s ease-in-out;
      }
    }
    
    @keyframes float {
      0% { transform: translateY(0px); }
      50% { transform: translateY(-20px); }
      100% { transform: translateY(0px); }
    }
    
    @keyframes fadeIn {
      from { opacity: 0; }
      to { opacity: 1; }
    }
    
    @keyframes slideUp {
      from { transform: translateY(30px); opacity: 0; }
      to { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes scaleIn {
      from { transform: scale(0.9); opacity: 0; }
      to { transform: scale(1); opacity: 1; }
    }
    
    @keyframes pulse-soft {
      0%, 100% { opacity: 0.9; }
      50% { opacity: 0.7; }
    }
    
    .pulse-soft {
      animation: pulse-soft 4s ease-in-out infinite;
    }
  </style>
</head>

<body class="font-sans min-h-screen overflow-x-hidden">
  <!-- 语言选择器 -->
  <div class="fixed top-4 right-4 z-40 bg-white/80 backdrop-blur-sm rounded-full shadow-lg p-1 flex items-center">
    <select id="languageSelector" class="bg-transparent border-none px-3 py-2 text-sm font-medium focus:outline-none focus:ring-0">
      <option value="zh">中文</option>
      <option value="en">English</option>
      <option value="ja">日本語</option>
      <option value="ko">한국어</option>
      <option value="fr">Français</option>
      <option value="de">Deutsch</option>
      <option value="ru">Русский</option>
    </select>
  </div>
  
  <!-- 背景图容器 -->
  <div class="fixed inset-0 -z-20">
    <div id="bgContainer" class="w-full h-full">
      <!-- 背景图将通过JS动态添加 -->
    </div>
  </div>
  
  <!-- 装饰元素 -->
  <div class="fixed inset-0 -z-10 overflow-hidden">
    <div class="absolute top-1/4 left-1/6 w-64 h-64 bg-primary/30 rounded-full blur-3xl animate-float"></div>
    <div class="absolute bottom-1/3 right-1/5 w-80 h-80 bg-secondary/30 rounded-full blur-3xl animate-float-delay"></div>
    <div class="absolute top-2/3 left-1/3 w-48 h-48 bg-accent/20 rounded-full blur-3xl animate-float-slow"></div>
  </div>
  
  <!-- 主界面 -->
  <div id="mainScreen" class="min-h-screen flex flex-col items-center justify-center p-4 sm:p-6 lg:p-8">
    <!-- 系统标题 -->
    <h1 class="text-[clamp(2rem,6vw,4rem)] font-bold text-gray-800 text-center mb-4 opacity-0 fade-in" style="animation-delay: 0.3s" data-i18n="title">
      智能心境评估系统
    </h1>
    
    <!-- 副标题 -->
    <p class="text-[clamp(1rem,2vw,1.25rem)] text-gray-600 text-center max-w-2xl mb-12 opacity-0 slide-up" style="animation-delay: 0.6s" data-i18n="subtitle">
      专业评估您的心理状态，提供个性化建议，陪伴您度过每一个时刻
    </p>
    
    <!-- 优势特点 -->
    <div id="featuresContainer" class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 max-w-7xl w-full mb-16">
      <!-- 特点将通过JS动态添加 -->
    </div>
    
    <!-- 立即登录按钮 -->
    <button id="loginButton" class="bg-gradient-custom text-gray-800 py-4 px-8 rounded-full font-semibold shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1 active:translate-y-0 text-lg opacity-0 scale-in" style="animation-delay: 1.8s">
      <span class="flex items-center">
        <span data-i18n="loginBtn">立即登录</span>
        <i class="fa fa-arrow-right ml-2 transform group-hover:translate-x-1 transition-transform duration-300"></i>
      </span>
    </button>
  </div>
  
  <!-- 登录卡片 (默认隐藏) -->
  <div id="loginScreen" class="fixed inset-0 bg-black/40 backdrop-blur-sm flex items-center justify-center p-4 sm:p-6 lg:p-8 z-50 hidden opacity-0 transition-opacity duration-300">
    <div class="w-full max-w-md transform scale-95 transition-transform duration-300">
      <div class="bg-white rounded-2xl shadow-2xl overflow-hidden">
        <!-- 卡片头部 -->
        <div class="bg-gradient-custom p-6 sm:p-8 relative overflow-hidden">
          <div class="absolute inset-0 bg-[url('https://picsum.photos/id/106/800/400')] opacity-10 bg-cover bg-center"></div>
          <div class="relative z-10">
            <h2 class="text-[clamp(1.5rem,3vw,2rem)] font-bold text-gray-800 text-center text-shadow" data-i18n="完善信息">
              请完善您的信息
            </h2>
            <p class="text-gray-700/80 text-center mt-2" data-i18n="填写信息提示">
              填写以下信息完成登录
            </p>
          </div>
          <div class="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-transparent via-gray-700/30 to-transparent animate-pulse"></div>
        </div>
        
        <!-- 表单区域 -->
        <div class="p-6 sm:p-8">
          <form id="loginForm" class="space-y-6">
            <!-- 关闭按钮 -->
            <div class="flex justify-end">
              <button type="button" id="closeLogin" class="text-gray-400 hover:text-gray-600 transition-colors">
                <i class="fa fa-times text-xl"></i>
              </button>
            </div>
            
            <!-- 性别选择 -->
            <div class="transform transition-all duration-300 hover:translate-x-1">
              <label class="block text-sm font-medium text-gray-700 mb-3" data-i18n="性别">
                性别 <span class="text-red-500">*</span>
              </label>
              <div class="grid grid-cols-2 gap-4">
                <label class="gender-option relative">
                  <input type="radio" name="gender" value="male" class="sr-only peer" required>
                  <div class="flex items-center justify-center p-5 border-2 rounded-xl cursor-pointer transition-all duration-300 peer-checked:border-accent peer-checked:bg-primary/10 hover:shadow-md">
                    <i class="fa fa-male text-gray-500 peer-checked:text-accent text-2xl"></i>
                    <span class="ml-3 text-gray-700 peer-checked:text-accent font-medium" data-i18n="男">男</span>
                  </div>
                  <div class="absolute -top-1 -right-1 w-5 h-5 bg-accent rounded-full flex items-center justify-center opacity-0 peer-checked:opacity-100 transition-all duration-300">
                    <i class="fa fa-check text-white text-xs"></i>
                  </div>
                </label>
                
                <label class="gender-option relative">
                  <input type="radio" name="gender" value="female" class="sr-only peer" required>
                  <div class="flex items-center justify-center p-5 border-2 rounded-xl cursor-pointer transition-all duration-300 peer-checked:border-accent peer-checked:bg-primary/10 hover:shadow-md">
                    <i class="fa fa-female text-gray-500 peer-checked:text-accent text-2xl"></i>
                    <span class="ml-3 text-gray-700 peer-checked:text-accent font-medium" data-i18n="女">女</span>
                  </div>
                  <div class="absolute -top-1 -right-1 w-5 h-5 bg-accent rounded-full flex items-center justify-center opacity-0 peer-checked:opacity-100 transition-all duration-300">
                    <i class="fa fa-check text-white text-xs"></i>
                  </div>
                </label>
              </div>
            </div>
            
            <!-- 职业选择（带搜索功能） -->
            <div class="transform transition-all duration-300 hover:translate-x-1">
              <label for="occupation" class="block text-sm font-medium text-gray-700 mb-3" data-i18n="职业">
                职业 <span class="text-red-500">*</span>
              </label>
              <div class="relative">
                <div class="relative mb-2">
                  <input type="text" id="occupationSearch" placeholder="搜索职业..." 
                    class="block w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary/50 focus:border-accent transition-all duration-300" data-i18n-placeholder="搜索职业">
                  <div class="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none text-gray-500">
                    <i class="fa fa-search"></i>
                  </div>
                </div>
                
                <select id="occupation" name="occupation" required
                  class="block w-full pl-4 pr-10 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary/50 focus:border-accent transition-all duration-300 appearance-none bg-white">
                  <option value="" disabled selected data-i18n="选择职业">请选择您的职业</option>
                  <!-- 职业选项会通过JS动态填充 -->
                </select>
                <div class="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-gray-500">
                  <i class="fa fa-chevron-down transition-transform duration-300" id="selectArrow"></i>
                </div>
                
                <!-- 职业列表提示 -->
                <div id="occupationList" class="absolute left-0 right-0 mt-1 max-h-60 overflow-y-auto bg-white border border-gray-300 rounded-lg shadow-lg z-10 hidden">
                  <ul class="py-1" id="occupationOptions">
                    <!-- 搜索结果会在这里显示 -->
                  </ul>
                </div>
              </div>
            </div>
            
            <!-- 年龄滚动选择 -->
            <div class="transform transition-all duration-300 hover:translate-x-1">
              <label class="block text-sm font-medium text-gray-700 mb-3" data-i18n="年龄">
                年龄 <span class="text-red-500">*</span>
                <span id="ageValue" class="ml-2 text-accent font-semibold text-lg">18</span>
                <span class="text-gray-500 text-sm ml-1" data-i18n="岁">岁</span>
              </label>
              <div class="px-2">
                <input type="range" id="age" name="age" min="1" max="120" value="18" 
                  class="w-full h-2 rounded-full bg-gray-200 appearance-none" oninput="updateAgeValue(this.value)">
              </div>
              <div class="flex justify-between text-xs text-gray-500 mt-1 px-1">
                <span data-i18n="1岁">1岁</span>
                <span data-i18n="120岁">120岁</span>
              </div>
            </div>
            
            <!-- 登录按钮 -->
            <div>
              <button type="submit" id="submitBtn"
                class="w-full bg-gradient-custom text-gray-800 py-3 px-4 rounded-xl font-medium shadow-md hover:shadow-lg transition-all duration-300 transform hover:-translate-y-1 active:translate-y-0 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:ring-offset-2 group">
                <span class="flex items-center justify-center">
                  <span data-i18n="完成登录">完成登录</span>
                  <i class="fa fa-arrow-right ml-2 transform group-hover:translate-x-1 transition-transform duration-300"></i>
                </span>
              </button>
            </div>
          </form>
        </div>
      </div>
      
      <!-- 页脚信息 -->
      <p class="text-center text-gray-500 text-sm mt-6 transform transition-all duration-300 hover:text-accent" data-i18n="登录条款">
        登录即表示您同意我们的<a href="#" class="text-accent hover:underline" data-i18n="服务条款">服务条款</a>和<a href="#" class="text-accent hover:underline" data-i18n="隐私政策">隐私政策</a>
      </p>
    </div>
  </div>
  
  <!-- 成功提示模态框 -->
  <div id="successModal" class="fixed inset-0 bg-black/50 flex items-center justify-center z-50 hidden opacity-0 transition-all duration-300">
    <div class="bg-white rounded-xl p-6 max-w-md w-full mx-4 transform scale-95 transition-all duration-300">
      <div class="text-center">
        <div class="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4 animate-bounce">
          <i class="fa fa-check text-2xl text-green-500"></i>
        </div>
        <h3 class="text-xl font-bold text-gray-900 mb-2" data-i18n="登录成功">登录成功！</h3>
        <p class="text-gray-600 mb-6" id="successMessage" data-i18n="跳转提示">正在为您跳转至智能心境评估系统...</p>
        <div class="w-full bg-gray-200 rounded-full h-2.5 mb-6">
          <div class="bg-primary h-2.5 rounded-full" id="progressBar" style="width: 0%"></div>
        </div>
      </div>
    </div>
  </div>

  <script>
    // 多语言文本数据
    const translations = {
      zh: {
        title: "智能心境评估系统",
        subtitle: "专业评估您的心理状态，提供个性化建议，陪伴您度过每一个时刻",
        loginBtn: "立即登录",
        "完善信息": "请完善您的信息",
        "填写信息提示": "填写以下信息完成登录",
        "性别": "性别",
        "男": "男",
        "女": "女",
        "职业": "职业",
        "搜索职业": "搜索职业...",
        "选择职业": "请选择您的职业",
        "年龄": "年龄",
        "岁": "岁",
        "1岁": "1岁",
        "120岁": "120岁",
        "完成登录": "完成登录",
        "登录条款": "登录即表示您同意我们的",
        "服务条款": "服务条款",
        "隐私政策": "隐私政策",
        "登录成功": "登录成功！",
        "跳转提示": "正在为您跳转至智能心境评估系统...",
        // 系统优势
        features: [
          { title: "智能评估", desc: "基于先进算法，精准分析您的心理状态，提供科学评估报告" },
          { title: "隐私保护", desc: "严格保护您的个人信息与评估结果，确保数据安全与隐私" },
          { title: "个性化建议", desc: "根据您的评估结果，提供量身定制的调整建议与放松方案" },
          { title: "趋势追踪", desc: "长期记录您的心境变化，生成详细趋势图表，直观了解状态" },
          { title: "适时提醒", desc: "根据您的习惯，智能提醒进行定期评估，保持心理健康监测" },
          { title: "专业咨询", desc: "提供与专业心理咨询师对接服务，必要时给予深度帮助" },
          { title: "音乐疗愈", desc: "根据心境推荐合适的疗愈音乐，帮助缓解压力与焦虑" },
          { title: "心理知识库", desc: "丰富的心理健康文章与指南，助您更好了解心理知识" },
          { title: "日程规划", desc: "结合您的心理状态，推荐合适的日常活动与休息计划" }
        ]
      },
      en: {
        title: "Intelligent Mood Assessment System",
        subtitle: "Professionally assess your mental state, provide personalized advice, and accompany you through every moment",
        loginBtn: "Login Now",
        "完善信息": "Please complete your information",
        "填写信息提示": "Fill in the following information to complete login",
        "性别": "Gender",
        "男": "Male",
        "女": "Female",
        "职业": "Occupation",
        "搜索职业": "Search occupation...",
        "选择职业": "Please select your occupation",
        "年龄": "Age",
        "岁": "years",
        "1岁": "1 year",
        "120岁": "120 years",
        "完成登录": "Complete Login",
        "登录条款": "Logging in means you agree to our",
        "服务条款": "Terms of Service",
        "隐私政策": "Privacy Policy",
        "登录成功": "Login successful!",
        "跳转提示": "Redirecting to the Intelligent Mood Assessment System...",
        // 系统优势
        features: [
          { title: "Intelligent Assessment", desc: "Based on advanced algorithms, accurately analyze your mental state and provide scientific assessment reports" },
          { title: "Privacy Protection", desc: "Strictly protect your personal information and assessment results to ensure data security and privacy" },
          { title: "Personalized Advice", desc: "Provide tailored adjustment suggestions and relaxation plans based on your assessment results" },
          { title: "Trend Tracking", desc: "Record your mood changes over time, generate detailed trend charts for intuitive understanding" },
          { title: "Timely Reminders", desc: "Intelligently remind you for regular assessments based on your habits to maintain mental health monitoring" },
          { title: "Professional Consultation", desc: "Provide access to professional counselors for in-depth help when needed" },
          { title: "Music Therapy", desc: "Recommend suitable healing music based on your mood to help relieve stress and anxiety" },
          { title: "Psychological Knowledge Base", desc: "Rich collection of mental health articles and guides to help you better understand psychological knowledge" },
          { title: "Schedule Planning", desc: "Recommend appropriate daily activities and rest plans based on your mental state" }
        ]
      }
    };
    
    // 切换语言函数
    function changeLanguage(lang) {
      // 更新所有文本内容
      document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        if (translations[lang] && translations[lang][key]) {
          el.textContent = translations[lang][key];
        }
      });
      
      // 更新占位符
      document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
        const key = el.getAttribute('data-i18n-placeholder');
        if (translations[lang] && translations[lang][key]) {
          el.placeholder = translations[lang][key];
        }
      });
      
      // 更新特点展示
      currentFeatureIndex = 0;
      displayFeatures();
    }
    
    // 背景图片数组 - 自然主题
    const backgroundImages = [
      'https://picsum.photos/id/10/1920/1080', // 山脉和森林
      'https://picsum.photos/id/15/1920/1080', // 海岸线
      'https://picsum.photos/id/29/1920/1080', // 森林小径
      'https://picsum.photos/id/42/1920/1080', // 河流和森林
      'https://picsum.photos/id/65/1920/1080', // 田野和云彩
      'https://picsum.photos/id/76/1920/1080', // 雪山
      'https://picsum.photos/id/87/1920/1080', // 湖泊和山脉
      'https://picsum.photos/id/96/1920/1080'  // 森林和阳光
    ];
    
    // 系统优势特点图标
    const featureIcons = [
      'fa-brain', 'fa-lock', 'fa-heart', 
      'fa-line-chart', 'fa-bell', 'fa-comments',
      'fa-music', 'fa-book', 'fa-calendar-check-o'
    ];
    
    // 职业数据
    const occupations = [
      "学生", "教师", "工程师", "医生", "护士", "律师", "会计师", 
      "程序员", "设计师", "建筑师", "记者", "编辑", "翻译", "厨师",
      "服务员", "司机", "警察", "消防员", "军人", "科学家", "研究员",
      "艺术家", "音乐家", "演员", "运动员", "商人", "企业家", "销售",
      "市场专员", "人力资源", "行政人员", "秘书", "顾问", "咨询师",
      "导游", "摄影师", "作家", "农民", "工人", "退休人员", "自由职业者"
    ];
    
    // 当前背景图索引
    let currentBgIndex = 0;
    // 当前显示的特点索引
    let currentFeatureIndex = 0;
    // 每次显示的特点数量（改为3个）
    const featuresPerPage = 3;
    
    // DOM元素
    const mainScreen = document.getElementById('mainScreen');
    const loginScreen = document.getElementById('loginScreen');
    const loginButton = document.getElementById('loginButton');
    const closeLogin = document.getElementById('closeLogin');
    const loginForm = document.getElementById('loginForm');
    const successModal = document.getElementById('successModal');
    const successMessage = document.getElementById('successMessage');
    const submitBtn = document.getElementById('submitBtn');
    const occupationSelect = document.getElementById('occupation');
    const occupationSearch = document.getElementById('occupationSearch');
    const occupationList = document.getElementById('occupationList');
    const occupationOptions = document.getElementById('occupationOptions');
    const selectArrow = document.getElementById('selectArrow');
    const progressBar = document.getElementById('progressBar');
    const bgContainer = document.getElementById('bgContainer');
    const featuresContainer = document.getElementById('featuresContainer');
    const languageSelector = document.getElementById('languageSelector');
    
    // 初始化背景图
    function initBackgrounds() {
      // 创建所有背景图元素
      backgroundImages.forEach((src, index) => {
        const img = document.createElement('img');
        img.src = src;
        img.alt = `背景图 ${index + 1}`;
        img.className = `absolute inset-0 w-full h-full object-cover bg-transition ${index === 0 ? 'opacity-20' : 'opacity-0'}`;
        img.style.zIndex = index === 0 ? '1' : '0';
        bgContainer.appendChild(img);
      });
      
      // 开始切换背景图
      setInterval(changeBackground, 8000);
    }
    
    // 切换背景图
    function changeBackground() {
      const images = bgContainer.querySelectorAll('img');
      const nextIndex = (currentBgIndex + 1) % backgroundImages.length;
      
      // 淡出当前图片
      images[currentBgIndex].style.opacity = '0';
      images[currentBgIndex].style.zIndex = '0';
      
      // 淡入下一张图片
      images[nextIndex].style.opacity = '0.2';
      images[nextIndex].style.zIndex = '1';
      
      currentBgIndex = nextIndex;
      
      // 同时更新显示的特点
      updateFeatures();
    }
    
    // 初始化特点展示
    function initFeatures() {
      displayFeatures();
    }
    
    // 显示特点
    function displayFeatures() {
      featuresContainer.innerHTML = '';
      const currentLang = languageSelector.value;
      const currentFeatures = translations[currentLang].features;
      
      // 计算当前要显示的特点范围
      const startIndex = currentFeatureIndex;
      let endIndex = currentFeatureIndex + featuresPerPage;
      
      // 如果超过总数则循环
      if (endIndex > currentFeatures.length) {
        endIndex = currentFeatures.length;
      }
      
      // 添加当前页的特点
      for (let i = startIndex; i < endIndex; i++) {
        const feature = currentFeatures[i];
        const featureEl = document.createElement('div');
        featureEl.className = 'bg-white/80 backdrop-blur-sm p-6 rounded-2xl shadow-lg transform transition-all duration-500 hover:shadow-xl hover:-translate-y-2 opacity-0 slide-up';
        featureEl.style.animationDelay = `${(i % featuresPerPage) * 0.2 + 0.3}s`;
        
        featureEl.innerHTML = `
          <div class="w-14 h-14 bg-primary/20 rounded-full flex items-center justify-center mb-4 transform transition-transform duration-300 hover:scale-110">
            <i class="fa ${featureIcons[i]} text-2xl text-accent"></i>
          </div>
          <h3 class="text-xl font-semibold text-gray-800 mb-2">${feature.title}</h3>
          <p class="text-gray-600">${feature.desc}</p>
        `;
        
        featuresContainer.appendChild(featureEl);
        
        // 触发动画
        setTimeout(() => {
          featureEl.classList.remove('opacity-0');
        }, 50);
      }
      
      // 更新当前索引，循环显示
      currentFeatureIndex = endIndex % currentFeatures.length;
    }
    
    // 更新显示的特点（随背景切换）
    function updateFeatures() {
      // 先添加淡出动画
      const currentFeatures = featuresContainer.querySelectorAll('div');
      currentFeatures.forEach((el, index) => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        
        // 延迟移除元素，等待动画完成
        if (index === currentFeatures.length - 1) {
          setTimeout(() => {
            displayFeatures();
          }, 500);
        }
      });
    }
    
    // 初始化职业选择器
    function initOccupations() {
      // 填充下拉列表
      occupations.forEach(occ => {
        const option = document.createElement('option');
        option.value = occ;
        option.textContent = occ;
        occupationSelect.appendChild(option);
      });
      
      // 显示所有职业选项
      function displayOccupations(list) {
        occupationOptions.innerHTML = '';
        if (list.length === 0) {
          const li = document.createElement('li');
          li.className = 'px-4 py-2 text-gray-500 hover:bg-gray-50 transition-colors';
          li.textContent = '没有找到匹配的职业';
          occupationOptions.appendChild(li);
          return;
        }
        
        list.forEach(occ => {
          const li = document.createElement('li');
          li.className = 'px-4 py-2 hover:bg-primary/10 cursor-pointer transition-all duration-200 hover:translate-x-1';
          li.textContent = occ;
          li.addEventListener('click', () => {
            occupationSelect.value = occ;
            occupationSearch.value = occ;
            occupationList.classList.add('hidden');
            selectArrow.classList.remove('rotate-180');
          });
          occupationOptions.appendChild(li);
        });
      }
      
      // 初始显示所有职业
      displayOccupations(occupations);
      
      // 搜索功能
      occupationSearch.addEventListener('input', (e) => {
        const searchTerm = e.target.value.toLowerCase();
        const filtered = occupations.filter(occ => 
          occ.toLowerCase().includes(searchTerm)
        );
        displayOccupations(filtered);
        occupationList.classList.remove('hidden');
      });
      
      // 下拉箭头点击事件
      selectArrow.addEventListener('click', () => {
        occupationList.classList.toggle('hidden');
        selectArrow.classList.toggle('rotate-180');
      });
      
      // 点击外部关闭下拉列表
      document.addEventListener('click', (e) => {
        if (!occupationList.contains(e.target) && 
            e.target !== occupationSearch && 
            e.target !== selectArrow) {
          occupationList.classList.add('hidden');
          selectArrow.classList.remove('rotate-180');
        }
      });
      
      // 选择框变化时同步搜索框
      occupationSelect.addEventListener('change', () => {
        occupationSearch.value = occupationSelect.value;
      });
    }
    
    // 更新年龄显示
    function updateAgeValue(value) {
      const ageValue = document.getElementById('ageValue');
      ageValue.textContent = value;
      
      // 添加数值变化动画
      ageValue.classList.add('scale-125', 'text-accent/80');
      setTimeout(() => {
        ageValue.classList.remove('scale-125', 'text-accent/80');
      }, 200);
    }
    
    // 显示登录界面
    function showLoginScreen() {
      loginScreen.classList.remove('hidden');
      setTimeout(() => {
        loginScreen.classList.add('opacity-100');
        loginScreen.querySelector('div').classList.remove('scale-95');
        loginScreen.querySelector('div').classList.add('scale-100');
      }, 10);
    }
    
    // 隐藏登录界面
    function hideLoginScreen() {
      loginScreen.classList.remove('opacity-100');
      loginScreen.querySelector('div').classList.remove('scale-100');
      loginScreen.querySelector('div').classList.add('scale-95');
      
      setTimeout(() => {
        loginScreen.classList.add('hidden');
      }, 300);
    }
    
    // 显示成功模态框并模拟跳转
    function showSuccessModal(message) {
      successMessage.textContent = message;
      successModal.classList.remove('hidden');
      setTimeout(() => {
        successModal.classList.add('opacity-100');
        successModal.querySelector('div').classList.remove('scale-95');
        successModal.querySelector('div').classList.add('scale-100');
        
        // 进度条动画
        let progress = 0;
        const interval = setInterval(() => {
          progress += 1;
          progressBar.style.width = `${progress}%`;
          
          if (progress >= 100) {
            clearInterval(interval);
            // 模拟跳转 - 直接跳转到应用页面
            setTimeout(() => {
              window.location.href = '/app';
            }, 500);
          }
        }, 30);
      }, 10);
    }
    
    // 隐藏成功模态框
    function hideSuccessModal() {
      successModal.classList.remove('opacity-100');
      successModal.querySelector('div').classList.remove('scale-100');
      successModal.querySelector('div').classList.add('scale-95');
      
      setTimeout(() => {
        successModal.classList.add('hidden');
        progressBar.style.width = '0%';
      }, 300);
    }
    
    // 表单提交处理
    loginForm.addEventListener('submit', function(e) {
      e.preventDefault();
      
      // 获取表单数据
      const formData = new FormData(this);
      const gender = formData.get('gender');
      const occupation = formData.get('occupation');
      const age = formData.get('age');
      
      // 显示加载状态
      submitBtn.disabled = true;
      submitBtn.innerHTML = '<i class="fa fa-spinner fa-spin"></i> ' + translations[languageSelector.value]["填写信息提示"].substring(0, 4) + '...';
      
      // 添加表单提交动画
      const formElements = this.querySelectorAll('input, select, label');
      formElements.forEach((el, index) => {
        setTimeout(() => {
          el.classList.add('opacity-50');
          setTimeout(() => el.classList.remove('opacity-50'), 300);
        }, index * 100);
      });
      
      // 模拟API请求延迟
      setTimeout(() => {
        const currentLang = languageSelector.value;
        // 构建成功消息
        const genderText = gender === 'male' ? translations[currentLang]["男"] : translations[currentLang]["女"];
        const message = `${translations[currentLang]["登录成功"].replace("！", "：")}${genderText}，${occupation}，${age}${translations[currentLang]["岁"]}。${translations[currentLang]["跳转提示"]}`;
        
        // 显示成功模态框
        showSuccessModal(message);
        
        // 重置按钮状态
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<span class="flex items-center justify-center"><span data-i18n="完成登录">完成登录</span><i class="fa fa-arrow-right ml-2 transform group-hover:translate-x-1 transition-transform duration-300"></i></span>';
      }, 1500);
    });
    
    // 事件监听
    loginButton.addEventListener('click', showLoginScreen);
    closeLogin.addEventListener('click', hideLoginScreen);
    languageSelector.addEventListener('change', (e) => changeLanguage(e.target.value));
    
    // 点击模态框外部关闭
    loginScreen.addEventListener('click', function(e) {
      if (e.target === loginScreen) {
        hideLoginScreen();
      }
    });
    
    successModal.addEventListener('click', function(e) {
      if (e.target === successModal) {
        hideSuccessModal();
      }
    });
    
    // 页面加载完成后初始化
    window.addEventListener('load', () => {
      initBackgrounds();
      initFeatures();
      initOccupations();
      
      // 自定义滑块样式
      const style = document.createElement('style');
      style.textContent = `
        input[type="range"]::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 24px;
          height: 24px;
          border-radius: 50%;
          background: #4ade80;
          border: 2px solid white;
          box-shadow: 0 0 0 1px rgba(5, 150, 105, 0.3);
          cursor: pointer;
          transition: all 0.2s ease;
        }
        
        input[type="range"]::-webkit-slider-thumb:hover {
          transform: scale(1.1);
          box-shadow: 0 0 0 3px rgba(5, 150, 105, 0.2);
        }
        
        input[type="range"]::-moz-range-thumb {
          width: 24px;
          height: 24px;
          border-radius: 50%;
          background: #4ade80;
          border: 2px solid white;
          box-shadow: 0 0 0 1px rgba(5, 150, 105, 0.3);
          cursor: pointer;
          transition: all 0.2s ease;
        }
        
        input[type="range"]::-moz-range-thumb:hover {
          transform: scale(1.1);
          box-shadow: 0 0 0 3px rgba(5, 150, 105, 0.2);
        }
      `;
      document.head.appendChild(style);
    });
  </script>
</body>
</html>
"""

# 路由定义
@app.get("/", response_class=HTMLResponse)
async def login_page():
    """登录页面"""
    return LOGIN_HTML

@app.get("/app")
async def app_page():
    """重定向到Gradio应用"""
    return RedirectResponse(url="/gradio")

# 创建Gradio应用
gradio_app = build_ui()

# 将Gradio应用挂载到FastAPI
app = gr.mount_gradio_app(app, gradio_app, path="/gradio")

if __name__ == "__main__":
    print("🚀 启动智能心境健康评估系统...")
    print("📱 登录界面: http://localhost:7860/")
    print("🧠 主应用界面: http://localhost:7860/app")
    
    uvicorn.run(app, host="0.0.0.0", port=7860)