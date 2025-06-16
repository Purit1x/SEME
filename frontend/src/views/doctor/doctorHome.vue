<!-- 
  doctorHome.vue
  医生主页组件
  功能：提供导航栏、用户信息展示和欢迎页面
  作者：同济大学软件工程管理与经济学项目组
  日期：2025
-->

<template>
  <el-container class="layout-container">
    <!-- 顶部导航栏 -->
    <el-header>
      <!-- 系统标题 -->
      <div class="header-left">
        <span class="system-title">前列腺消融手术预后疗效预测系统</span>
      </div>

      <!-- 导航菜单 -->
      <div class="header-center">
        <el-menu
          :default-active="activeMenu"
          mode="horizontal"
          router
          class="nav-menu"
        >
          <el-menu-item
            v-for="menu in menuItems"
            :key="menu.path"
            :index="menu.path"
          >
            <el-icon><component :is="menu.icon" /></el-icon>
            {{ menu.title }}
          </el-menu-item>
        </el-menu>
      </div>

      <!-- 用户信息和下拉菜单 -->
      <div class="header-right">
        <el-dropdown @command="handleCommand">
          <span class="el-dropdown-link">
            <el-avatar :size="32" :src="doctorInfo.avatar || defaultAvatar" />
            {{ doctorInfo.name }}
            <el-icon><ArrowDown /></el-icon>
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
      <!-- 欢迎卡片 -->
      <el-card v-if="showWelcome" class="welcome-card">
        <template #header>
          <div class="welcome-header">
            <span>欢迎回来，{{ doctorInfo.name }} 医生</span>
            <el-button
              type="text"
              class="close-button"
              @click="showWelcome = false"
            >
              <el-icon><Close /></el-icon>
            </el-button>
          </div>
        </template>
        <div class="welcome-content">
          <p>您可以通过上方导航栏访问以下功能：</p>
          <ul>
            <li v-for="menu in menuItems" :key="menu.path">
              <b>{{ menu.title }}：</b>{{ menu.description }}
            </li>
          </ul>
        </div>
      </el-card>

      <!-- 路由视图 -->
      <router-view v-slot="{ Component }">
        <transition name="fade" mode="out-in">
          <component :is="Component" />
        </transition>
      </router-view>
    </el-main>
  </el-container>
</template>

<script setup>
import { ref, onMounted } from "vue";
import { useRouter } from "vue-router";
import { ElMessage, ElMessageBox } from "element-plus";
import {
  User,
  DataLine,
  Document,
  ArrowDown,
  Close,
} from "@element-plus/icons-vue";

// 路由实例
const router = useRouter();

// 常量和响应式数据
const defaultAvatar =
  "https://cube.elemecdn.com/3/7c/3ea6beec64369c2642b92c6726f1epng.png";
const activeMenu = ref("/patient-management");
const showWelcome = ref(true);
const doctorInfo = ref({});

// 导航菜单配置
const menuItems = [
  {
    path: "/patient-management",
    title: "患者管理",
    icon: "User",
    description: "管理患者信息和病历",
  },
  {
    path: "/effect-prediction",
    title: "效果预测",
    icon: "DataLine",
    description: "进行手术预后效果预测",
  },
  {
    path: "/prediction-records",
    title: "预测记录",
    icon: "Document",
    description: "查看历史预测记录",
  },
];

// 生命周期钩子
onMounted(() => {
  // 初始化医生信息
  initDoctorInfo();
  // 默认导航到患者管理页面
  router.push("/patient-management");
});

// 初始化医生信息
const initDoctorInfo = () => {
  const storedInfo = localStorage.getItem("doctorInfo");
  if (storedInfo) {
    doctorInfo.value = JSON.parse(storedInfo);
  }
};

// 处理下拉菜单命令
const handleCommand = async (command) => {
  if (command === "profile") {
    router.push("/doctor-profile");
    return;
  }

  if (command === "logout") {
    try {
      await ElMessageBox.confirm("确定要退出登录吗？", "提示", {
        confirmButtonText: "确定",
        cancelButtonText: "取消",
        type: "warning",
      })
      [
        // 清除本地存储
        ("access_token", "refresh_token", "doctorInfo")
      ].forEach((key) => localStorage.removeItem(key));

      // 跳转到登录页
      router.replace("/login");
      ElMessage.success("已退出登录");
    } catch {
      // 用户取消操作，不做处理
    }
  }
};
</script>

<style scoped>
/* 布局容器 */
.layout-container {
  height: 100vh;
  display: flex;
  flex-direction: column;
}

/* 头部样式 */
.el-header {
  background-color: #fff;
  display: flex;
  justify-content: space-between;
  align-items: center;
  box-shadow: 0 1px 4px rgba(0, 21, 41, 0.08);
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

/* 导航菜单样式 */
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
}

/* 用户信息区域样式 */
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

  &:hover {
    color: #409eff;
  }
}

.el-avatar {
  margin-right: 8px;
}

/* 主要内容区样式 */
.el-main {
  background-color: #f0f2f5;
  padding: 20px;
  flex: 1;
  overflow-y: auto;
}

/* 欢迎卡片样式 */
.welcome-card {
  margin-bottom: 20px;

  .welcome-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .welcome-content {
    color: #606266;

    ul {
      margin-top: 10px;
      padding-left: 20px;
    }

    li {
      margin-bottom: 8px;
    }
  }
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
