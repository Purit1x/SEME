<template>
    <div class="manager-home">
        <el-container>
            <el-aside width="200px">
                <div class="sidebar">
                    <div class="logo">
                        <h3>管理后台</h3>
                    </div>
                    <el-menu
                        :default-active="activeMenu"
                        class="sidebar-menu"
                        background-color="#001529"
                        text-color="#fff"
                        active-text-color="#409EFF"
                        router
                    >
                        <el-menu-item index="/manager/home">
                            <el-icon><HomeFilled /></el-icon>
                            <span>首页</span>
                        </el-menu-item>
                        <el-menu-item index="/manager/patients">
                            <el-icon><User /></el-icon>
                            <span>患者管理</span>
                        </el-menu-item>
                        <el-menu-item index="/manager/doctors">
                            <el-icon><UserFilled /></el-icon>
                            <span>医生管理</span>
                        </el-menu-item>
                    </el-menu>
                </div>
            </el-aside>
            <el-container>
                <el-header>
                    <div class="header-right">
                        <el-dropdown @command="handleCommand">
                            <span class="user-info">
                                <el-icon><User /></el-icon>
                                管理员
                            </span>
                            <template #dropdown>
                                <el-dropdown-menu>
                                    <el-dropdown-item command="logout">退出登录</el-dropdown-item>
                                </el-dropdown-menu>
                            </template>
                        </el-dropdown>
                    </div>
                </el-header>
                <el-main>
                    <router-view></router-view>
                </el-main>
            </el-container>
        </el-container>
    </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { HomeFilled, User, UserFilled } from '@element-plus/icons-vue'

const router = useRouter()
const route = useRoute()

const activeMenu = computed(() => route.path)

const handleCommand = (command) => {
    if (command === 'logout') {
        localStorage.removeItem('access_token')
        localStorage.removeItem('user_type')
        ElMessage.success('已退出登录')
        router.push('/manager/login')
    }
}
</script>

<style scoped>
.manager-home {
    height: 100vh;
}

.el-container {
    height: 100%;
}

.sidebar {
    height: 100%;
    background-color: #001529;
    color: white;
}

.logo {
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.logo h3 {
    margin: 0;
    color: white;
    font-size: 18px;
}

.sidebar-menu {
    border-right: none;
}

.el-header {
    background-color: white;
    border-bottom: 1px solid #eee;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding: 0 20px;
}

.header-right {
    display: flex;
    align-items: center;
}

.user-info {
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
    color: #666;
}

.el-main {
    background-color: #f0f2f5;
    padding: 20px;
}

.welcome-message {
    text-align: center;
    padding: 100px 0;
    color: #666;
}

.welcome-message h2 {
    font-size: 24px;
    font-weight: normal;
}
</style> 