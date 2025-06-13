<!--
  ManagerDashboard.vue
  管理后台仪表盘组件
  功能：展示系统概览、统计数据和快速访问入口
  作者：同济大学软件工程管理与经济学项目组
  日期：2025
-->

<template>
  <div class="manager-dashboard">
    <!-- 欢迎信息区域 -->
    <div class="welcome-section">
      <h2>欢迎使用MRI预测系统管理后台</h2>
      <p>在这里您可以管理系统的各项功能和查看数据统计</p>
    </div>

    <!-- 统计卡片区域 -->
    <div class="statistics-cards">
      <!-- 用户统计卡片 -->
      <el-card class="stat-card" shadow="hover">
        <template #header>
          <div class="card-header">
            <el-icon><User /></el-icon>
            <span>用户统计</span>
          </div>
        </template>
        <div class="card-content">
          <div class="stat-item">
            <span class="label">总医生数</span>
            <span class="value">{{ statistics.doctorCount }}</span>
          </div>
          <div class="stat-item">
            <span class="label">总患者数</span>
            <span class="value">{{ statistics.patientCount }}</span>
          </div>
        </div>
      </el-card>

      <!-- 预测统计卡片 -->
      <el-card class="stat-card" shadow="hover">
        <template #header>
          <div class="card-header">
            <el-icon><DataLine /></el-icon>
            <span>预测统计</span>
          </div>
        </template>
        <div class="card-content">
          <div class="stat-item">
            <span class="label">今日预测次数</span>
            <span class="value">{{ statistics.todayPredictions }}</span>
          </div>
          <div class="stat-item">
            <span class="label">总预测次数</span>
            <span class="value">{{ statistics.totalPredictions }}</span>
          </div>
        </div>
      </el-card>

      <!-- 系统状态卡片 -->
      <el-card class="stat-card" shadow="hover">
        <template #header>
          <div class="card-header">
            <el-icon><Monitor /></el-icon>
            <span>系统状态</span>
          </div>
        </template>
        <div class="card-content">
          <div class="stat-item">
            <span class="label">系统运行时间</span>
            <span class="value">{{ statistics.uptime }}</span>
          </div>
          <div class="stat-item">
            <span class="label">模型版本</span>
            <span class="value">{{ statistics.modelVersion }}</span>
          </div>
        </div>
      </el-card>
    </div>

    <!-- 快速操作区域 -->
    <div class="quick-actions">
      <h3 class="section-title">快速操作</h3>
      <div class="action-buttons">
        <el-button type="primary" @click="navigateTo('/manager/doctors')">
          <el-icon><UserFilled /></el-icon>医生管理
        </el-button>
        <el-button type="success" @click="navigateTo('/manager/patients')">
          <el-icon><User /></el-icon>患者管理
        </el-button>
        <el-button type="warning" @click="refreshStatistics">
          <el-icon><Refresh /></el-icon>刷新数据
        </el-button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from "vue";
import { useRouter } from "vue-router";
import {
  User,
  UserFilled,
  DataLine,
  Monitor,
  Refresh,
} from "@element-plus/icons-vue";

// 路由实例
const router = useRouter();

/**
 * 统计数据对象
 * @typedef {Object} Statistics
 * @property {number} doctorCount - 医生总数
 * @property {number} patientCount - 患者总数
 * @property {number} todayPredictions - 今日预测次数
 * @property {number} totalPredictions - 总预测次数
 * @property {string} uptime - 系统运行时间
 * @property {string} modelVersion - 模型版本
 */

// 统计数据
const statistics = ref({
  doctorCount: 0,
  patientCount: 0,
  todayPredictions: 0,
  totalPredictions: 0,
  uptime: "0天0小时",
  modelVersion: "v1.0.0",
});

/**
 * 页面导航函数
 * @param {string} path - 目标路由路径
 */
const navigateTo = (path) => {
  router.push(path);
};

/**
 * 刷新统计数据
 * 从后端API获取最新的统计信息
 */
const refreshStatistics = async () => {
  try {
    // TODO: 调用后端API获取最新统计数据
    // const response = await fetch('/api/statistics')
    // statistics.value = await response.json()

    // 模拟数据更新
    statistics.value = {
      doctorCount: Math.floor(Math.random() * 100),
      patientCount: Math.floor(Math.random() * 1000),
      todayPredictions: Math.floor(Math.random() * 50),
      totalPredictions: Math.floor(Math.random() * 5000),
      uptime: "7天12小时",
      modelVersion: "v1.0.0",
    };
  } catch (error) {
    console.error("获取统计数据失败:", error);
  }
};

// 组件挂载时获取初始数据
onMounted(() => {
  refreshStatistics();
});
</script>

<style scoped>
/* 仪表盘整体布局 */
.manager-dashboard {
  padding: 24px;
  background-color: #f5f7fa;
  min-height: calc(100vh - 120px);
}

/* 欢迎区域样式 */
.welcome-section {
  text-align: center;
  padding: 40px 0;
  margin-bottom: 40px;
  background: linear-gradient(135deg, #1890ff 0%, #36cfc9 100%);
  border-radius: 8px;
  color: white;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

.welcome-section h2 {
  font-size: 28px;
  font-weight: 600;
  margin-bottom: 16px;
}

.welcome-section p {
  font-size: 16px;
  opacity: 0.9;
}

/* 统计卡片区域样式 */
.statistics-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 24px;
  margin-bottom: 40px;
}

.stat-card {
  transition: all 0.3s ease;
}

.stat-card:hover {
  transform: translateY(-5px);
}

.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 16px;
  font-weight: 600;
}

.card-content {
  padding: 20px 0;
}

.stat-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.stat-item:last-child {
  margin-bottom: 0;
}

.stat-item .label {
  color: #909399;
  font-size: 14px;
}

.stat-item .value {
  font-size: 20px;
  font-weight: 600;
  color: #303133;
}

/* 快速操作区域样式 */
.quick-actions {
  background-color: white;
  padding: 24px;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

.section-title {
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 20px;
  color: #303133;
}

.action-buttons {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}

.action-buttons .el-button {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 24px;
}

/* 响应式布局调整 */
@media (max-width: 768px) {
  .statistics-cards {
    grid-template-columns: 1fr;
  }

  .action-buttons {
    flex-direction: column;
  }

  .action-buttons .el-button {
    width: 100%;
  }
}
</style> 