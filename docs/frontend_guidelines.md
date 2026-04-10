# 内部前端开发规范

本文档定义了 DevMate 项目的前端开发标准，所有生成的前端项目必须遵守以下规范。

## 通用原则

- 使用语义化 HTML5 标签（`<header>`, `<nav>`, `<main>`, `<section>`, `<article>`, `<footer>`）
- 所有页面必须支持响应式布局，适配桌面（>1024px）、平板（768-1024px）、手机（<768px）
- 优先使用 CSS Flexbox 和 Grid 进行布局
- 支持 `lang="zh-CN"` 中文本地化
- 字符编码统一使用 UTF-8

## 项目结构

```
project-name/
├── index.html
├── css/
│   └── style.css
├── js/
│   └── app.js
└── assets/
    └── images/
```

- HTML、CSS、JS 必须分文件存放，严禁内联样式和脚本（微小交互除外）
- 静态资源统一放在 `assets/` 目录下

## HTML 规范

- 文档类型声明：`<!DOCTYPE html>`
- 必须包含 viewport meta 标签：`<meta name="viewport" content="width=device-width, initial-scale=1.0">`
- 使用 `<meta charset="UTF-8">` 声明编码
- 页面必须包含 `<title>` 标签
- 外部资源使用 CDN 链接（如 Tailwind CSS、Font Awesome、Leaflet 等）
- 图片必须包含 `alt` 属性和合理的宽高设置
- 表单元素必须关联 `<label>`

## CSS 规范

### 配色方案

使用 CSS 自定义属性（CSS Variables）定义主题配色：

```css
:root {
    --primary-color: #2d6a4f;
    --primary-light: #40916c;
    --primary-dark: #1b4332;
    --secondary-color: #95d5b2;
    --accent-color: #ff6b35;
    --text-color: #333333;
    --text-light: #666666;
    --bg-color: #ffffff;
    --bg-light: #f8f9fa;
    --border-color: #e0e0e0;
}
```

- 主色调以自然/户外风格为主（绿色系）
- 强调色用于按钮、链接、重要提示
- 文字与背景对比度至少 4.5:1（WCAG AA 标准）

### 布局

- 最大内容宽度：1200px，居中显示
- 使用 `max-width` 而非固定 `width`
- 间距使用 8px 的倍数（8px, 16px, 24px, 32px, 48px, 64px）
- 卡片圆角：8px-12px
- 阴影层次：`box-shadow: 0 2px 8px rgba(0,0,0,0.1)` 为标准卡片阴影

### 响应式断点

```css
/* 手机 */
@media (max-width: 767px) { ... }
/* 平板 */
@media (min-width: 768px) and (max-width: 1023px) { ... }
/* 桌面 */
@media (min-width: 1024px) { ... }
```

- 移动端导航栏折叠为汉堡菜单
- 卡片布局：桌面 3 列、平板 2 列、手机 1 列
- 字体大小移动端适当缩小

### 动画

- 过渡时间：`transition: all 0.3s ease`
- 卡片悬浮效果：轻微上移 + 阴影增强
- 避免过度动画，保持页面流畅

## JavaScript 规范

- 使用 ES6+ 语法（`const`/`let`、箭头函数、模板字符串、解构赋值）
- 使用 `addEventListener` 绑定事件，禁止 `onclick` 内联
- 异步操作使用 `async/await`，配合 `try/catch` 错误处理
- DOM 操作使用 `document.querySelector` / `querySelectorAll`
- 数据与视图分离，使用状态对象管理数据
- 交互反馈：使用 Toast 提示替代 `alert()`
- 节流/防抖：搜索输入和滚动事件必须做防抖处理

```javascript
// 防抖函数示例
function debounce(fn, delay = 300) {
    let timer;
    return function (...args) {
        clearTimeout(timer);
        timer = setTimeout(() => fn.apply(this, args), delay);
    };
}
```

## 常用组件规范

### 导航栏

- 固定定位（`position: fixed`），滚动时添加阴影
- 包含 Logo、导航链接、搜索框（可选）
- 移动端使用汉堡菜单，点击展开/收起

### 卡片组件

- 包含：图片、标题、描述、标签、操作按钮
- 悬浮效果：`transform: translateY(-4px)` + 阴影增强
- 图片使用 `object-fit: cover` 保持比例
- 标签使用小号圆角背景

### 搜索框

- 输入框带搜索图标
- 支持回车搜索
- 搜索结果实时过滤或跳转

### 弹窗/模态框

- 半透明背景遮罩
- 居中显示，带关闭按钮
- 点击遮罩关闭
- 支持 ESC 键关闭

### 页脚

- 深色背景，浅色文字
- 多栏布局：品牌信息、导航链接、联系方式
- 社交媒体图标

## 地图集成（如需要）

- 使用 Leaflet.js（开源、轻量）
- 地图容器需指定高度
- 标记点使用自定义图标
- 弹窗展示详细信息

```html
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
```

## 图标

- 使用 Font Awesome 或 Lucide Icons
- 通过 CDN 引入
- 图标与文字对齐，适当间距

## 性能要求

- 图片使用懒加载：`loading="lazy"`
- 外部脚本使用 `defer` 属性
- CSS 放 `<head>` 中，JS 放 `</body>` 前
- 避免阻塞渲染的大型内联数据

## 无障碍（Accessibility）

- 所有交互元素可通过键盘访问（Tab 导航）
- 按钮和链接使用描述性文字
- 颜色对比度满足 WCAG AA 标准
- 图片提供 `alt` 文本
- 表单输入提供 `aria-label`
