<!--
  ManagerHome.vue
  管理后台主页面组件
  功能：提供管理后台的整体布局、导航菜单和用户操作界面
  作者：同济大学软件工程管理与经济学项目组
  日期：2025
-->

<template>
  <div class="manager-home">
    <!-- 整体布局容器 -->
    <el-container>
      <!-- 侧边栏 -->
      <el-aside width="200px">
        <div class="sidebar">
          <!-- Logo区域 -->
          <div class="logo">
            <h3>MRI影像管理系统</h3>
          </div>

          <!-- 导航菜单 -->
          <el-menu
            :default-active="activeMenu"
            class="sidebar-menu"
            background-color="#001529"
            text-color="#fff"
            active-text-color="#409EFF"
            :router="true"
            :unique-opened="true"
          >
            <!-- 首页 -->
            <el-menu-item index="/manager/home">
              <el-icon><HomeFilled /></el-icon>
              <template #title>首页</template>
            </el-menu-item>

            <!-- 患者管理 -->
            <el-menu-item index="/manager/patients">
              <el-icon><User /></el-icon>
              <template #title>患者管理</template>
            </el-menu-item>

            <!-- 医生管理 -->
            <el-menu-item index="/manager/doctors">
              <el-icon><UserFilled /></el-icon>
              <template #title>医生管理</template>
            </el-menu-item>
          </el-menu>
        </div>
      </el-aside>

      <!-- 主体区域 -->
      <el-container>
        <!-- 顶部导航栏 -->
        <el-header height="60px">
          <div class="header-content">
            <!-- 左侧面包屑 -->
            <div class="breadcrumb">
              <el-breadcrumb separator="/">
                <el-breadcrumb-item :to="{ path: '/manager/home' }"
                  >首页</el-breadcrumb-item
                >
                <el-breadcrumb-item v-if="route.meta.title">{{
                  route.meta.title
                }}</el-breadcrumb-item>
              </el-breadcrumb>
            </div>

            <!-- 右侧用户信息 -->
            <div class="header-right">
              <el-dropdown @command="handleCommand" trigger="click">
                <span class="user-info">
                  <el-avatar :size="32" icon="UserFilled" />
                  <span class="username">管理员</span>
                  <el-icon class="el-icon--right"><arrow-down /></el-icon>
                </span>
                <template #dropdown>
                  <el-dropdown-menu>
                    <el-dropdown-item command="logout">
                      <el-icon><switch-button /></el-icon>
                      退出登录
                    </el-dropdown-item>
                  </el-dropdown-menu>
                </template>
              </el-dropdown>
            </div>
          </div>
        </el-header>

        <!-- 主要内容区域 -->
        <el-main>
          <router-view v-slot="{ Component }">
            <transition name="fade" mode="out-in">
              <component :is="Component" />
            </transition>
          </router-view>
        </el-main>
      </el-container>
    </el-container>
  </div>
</template>

<script setup>
import { ref, computed } from "vue";
import { useRouter, useRoute } from "vue-router";
import { ElMessage } from "element-plus";
import {
  HomeFilled,
  User,
  UserFilled,
  ArrowDown,
  SwitchButton,
} from "@element-plus/icons-vue";

// 路由实例
const router = useRouter();
const route = useRoute();

/**
 * 计算当前激活的菜单项
 * 根据当前路由路径自动高亮对应的菜单
 */
const activeMenu = computed(() => route.path);

/**
 * 处理用户下拉菜单命令
 * @param {string} command - 菜单命令
 */
const handleCommand = (command) => {
  if (command === "logout") {
    // 清除登录信息
    localStorage.removeItem("access_token");
    localStorage.removeItem("user_type");

    // 提示用户
    ElMessage({
      type: "success",
      message: "已安全退出系统",
      duration: 2000,
    });

    // 跳转到登录页
    router.push("/manager/login");
  }
};
</script>

<style scoped>
/* 页面整体布局 */
.manager-home {
  height: 100vh;
  background-color: #f0f2f5;
}

.el-container {
  height: 100%;
}

/* 侧边栏样式 */
.sidebar {
  height: 100%;
  background-color: #001529;
  color: white;
  transition: all 0.3s ease;
}

/* Logo样式 */
.logo {
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  background-color: #002140;
}

.logo h3 {
  margin: 0;
  color: white;
  font-size: 18px;
  font-weight: 600;
  letter-spacing: 1px;
}

/* 导航菜单样式 */
.sidebar-menu {
  border-right: none;
  user-select: none;
}

.sidebar-menu :deep(.el-menu-item) {
  height: 50px;
  line-height: 50px;
  margin: 4px 0;
}

.sidebar-menu :deep(.el-menu-item.is-active) {
  background-color: #1890ff !important;
}

/* 顶部导航栏样式 */
.el-header {
  background-color: white;
  border-bottom: 1px solid #f0f0f0;
  box-shadow: 0 1px 4px rgba(0, 21, 41, 0.08);
  padding: 0;
}

.header-content {
  height: 100%;
  padding: 0 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.breadcrumb {
  font-size: 14px;
}

.header-right {
  display: flex;
  align-items: center;
}

/* 用户信息样式 */
.user-info {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  padding: 0 8px;
  height: 40px;
  border-radius: 4px;
  transition: all 0.3s;
}

.user-info:hover {
  background: rgba(0, 0, 0, 0.025);
}

.username {
  font-size: 14px;
  color: #666;
  margin: 0 4px;
}

/* 主要内容区域样式 */
.el-main {
  background-color: #f0f2f5;
  padding: 20px;
  min-height: calc(100vh - 60px);
}

/* 路由切换动画 */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style> 