<template>
    <el-container class="layout-container">
        <!-- 顶部导航栏 -->
        <el-header>
            <div class="header-left">
                <span class="system-title">前列腺消融手术预后疗效预测系统</span>
            </div>
            <div class="header-center">
                <el-menu
                    :default-active="activeMenu"
                    mode="horizontal"
                    router
                    class="nav-menu"
                >
                    <el-menu-item index="/patient-management">
                        <el-icon><User /></el-icon>
                        患者管理
                    </el-menu-item>
                    <el-menu-item index="/effect-prediction">
                        <el-icon><DataLine /></el-icon>
                        效果预测
                    </el-menu-item>
                    <el-menu-item index="/prediction-records">
                        <el-icon><Document /></el-icon>
                        预测记录
                    </el-menu-item>
                </el-menu>
            </div>
            <div class="header-right">
                <el-dropdown @command="handleCommand">
                    <span class="el-dropdown-link">
                        <el-avatar :size="32" :src="doctorInfo.avatar || defaultAvatar" />
                        {{ doctorInfo.name }}
                        <el-icon class="el-icon--right"><ArrowDown /></el-icon>
                    </span>
                    <template #dropdown>
                        <el-dropdown-menu>
                            <el-dropdown-item command="profile">个人信息</el-dropdown-item>
                            <el-dropdown-item command="logout">退出登录</el-dropdown-item>
                        </el-dropdown-menu>
                    </template>
                </el-dropdown>
            </div>
        </el-header>

        <!-- 主要内容区 -->
        <el-main>
            <!-- 欢迎信息卡片，只在首次登录时显示 -->
            <el-card v-if="showWelcome" class="welcome-card">
                <template #header>
                    <div class="welcome-header">
                        <span>欢迎回来，{{ doctorInfo.name }} 医生</span>
                        <el-button
                            type="text"
                            class="close-button"
                            @click="closeWelcome"
                        >
                            <el-icon><Close /></el-icon>
                        </el-button>
                    </div>
                </template>
                <div class="welcome-content">
                    <p>您可以通过上方导航栏访问以下功能：</p>
                    <ul>
                        <li><b>患者管理：</b>管理患者信息和病历</li>
                        <li><b>效果预测：</b>进行手术预后效果预测</li>
                        <li><b>预测记录：</b>查看历史预测记录</li>
                    </ul>
                </div>
            </el-card>

            <!-- 路由视图，用于显示各个功能页面 -->
            <router-view v-slot="{ Component }">
                <transition name="fade" mode="out-in">
                    <component :is="Component" />
                </transition>
            </router-view>
        </el-main>
    </el-container>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { User, DataLine, Document, ArrowDown, Close } from '@element-plus/icons-vue'

const router = useRouter()
const activeMenu = ref('/patient-management')
const defaultAvatar = 'https://cube.elemecdn.com/3/7c/3ea6beec64369c2642b92c6726f1epng.png'
const showWelcome = ref(true)

// 获取医生信息
const doctorInfo = ref({})
onMounted(() => {
    const storedInfo = localStorage.getItem('doctorInfo')
    if (storedInfo) {
        doctorInfo.value = JSON.parse(storedInfo)
    }
    // 默认跳转到患者管理页面
    router.push('/patient-management')
})

// 关闭欢迎卡片
const closeWelcome = () => {
    showWelcome.value = false
}

// 处理下拉菜单命令
const handleCommand = (command) => {
    switch (command) {
        case 'profile':
            router.push('/doctor-profile')
            break
        case 'logout':
            ElMessageBox.confirm(
                '确定要退出登录吗？',
                '提示',
                {
                    confirmButtonText: '确定',
                    cancelButtonText: '取消',
                    type: 'warning',
                }
            ).then(() => {
                localStorage.removeItem('access_token')
                localStorage.removeItem('refresh_token')
                localStorage.removeItem('doctorInfo')
                router.replace('/login')
                ElMessage.success('已退出登录')
            }).catch(() => {})
            break
    }
}
</script>

<style scoped>
.layout-container {
    height: 100vh;
    display: flex;
    flex-direction: column;
}

.el-header {
    background-color: #fff;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 1px 4px rgba(0,21,41,.08);
    padding: 0 20px;
    height: 60px;
    min-width: 1000px;
}

.header-left {
    width: 300px;
    flex-shrink: 0;
}

.system-title {
    font-size: 18px;
    font-weight: bold;
    color: #304156;
}

.header-center {
    flex: 1;
    display: flex;
    justify-content: center;
    min-width: 400px;
    margin: 0 20px;
}

.nav-menu {
    border-bottom: none;
    display: flex;
    justify-content: center;
    gap: 20px;
    width: 100%;
    max-width: 500px;
}

.nav-menu .el-menu-item {
    flex: 0 0 auto;
    height: 60px;
    line-height: 60px;
    padding: 0 20px;
    font-size: 16px;
    white-space: nowrap;
}

.header-right {
    width: 200px;
    flex-shrink: 0;
    display: flex;
    justify-content: flex-end;
}

.el-dropdown-link {
    display: flex;
    align-items: center;
    cursor: pointer;
    color: #606266;
}

.el-dropdown-link:hover {
    color: #409EFF;
}

.el-avatar {
    margin-right: 8px;
}

.el-main {
    background-color: #f0f2f5;
    padding: 20px;
    flex: 1;
    overflow-y: auto;
}

.welcome-card {
    margin-bottom: 20px;
}

.welcome-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.welcome-content {
    color: #606266;
}

.welcome-content ul {
    margin-top: 10px;
    padding-left: 20px;
}

.welcome-content li {
    margin-bottom: 8px;
}

.close-button {
    padding: 2px;
}

/* 页面切换动画 */
.fade-enter-active,
.fade-leave-active {
    transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
    opacity: 0;
}
</style>
