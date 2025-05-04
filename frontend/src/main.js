import { createApp } from 'vue'

import App from './App.vue'

import ElementPlus from 'element-plus'
import zhCn from 'element-plus/dist/locale/zh-cn.mjs'
import 'element-plus/dist/index.css'
import store from '@/stores/index.js'

import {router} from './router'   // 从export default router 改为 export function router = createRouter({}) 需要添加 {}

// 导入样式
import './assets/doctor/css/style.css'

const app = createApp(App)

// 状态管理
app.use(store)

// 路由管理
app.use(router)


app.use(ElementPlus, {
    locale: zhCn,
})

app.mount('#app')