<!--
  DoctorManagement.vue
  医生管理页面组件
  功能：提供医生列表管理、搜索、封禁/解封等功能
  作者：同济大学软件工程管理与经济学项目组
  日期：2025
-->

<template>
  <div class="doctor-management">
    <!-- 医生列表卡片 -->
    <el-card class="doctor-list">
      <template #header>
        <div class="card-header">
          <span>医生列表</span>
        </div>
      </template>

      <!-- 搜索栏 -->
      <div class="search-bar">
        <el-input
          v-model="searchQuery"
          placeholder="搜索医生姓名/工号/科室"
          clearable
          @clear="handleSearch"
          @input="handleSearch"
        >
          <template #prefix>
            <el-icon><Search /></el-icon>
          </template>
        </el-input>
      </div>

      <!-- 医生列表表格 -->
      <el-table
        :data="filteredDoctorList"
        style="width: 100%"
        v-loading="loading"
        border
        stripe
        highlight-current-row
      >
        <!-- 基本信息列 -->
        <el-table-column prop="doctor_id" label="工号" width="120" sortable />
        <el-table-column prop="name" label="姓名" width="120" />
        <el-table-column prop="department" label="科室" width="120" />
        <el-table-column
          prop="email"
          label="邮箱"
          min-width="200"
          show-overflow-tooltip
        />

        <!-- 状态列 -->
        <el-table-column label="状态" width="100" align="center">
          <template #default="{ row }">
            <el-tag
              :type="row.is_locked ? 'danger' : 'success'"
              size="small"
              effect="dark"
            >
              {{ row.is_locked ? "已封禁" : "正常" }}
            </el-tag>
          </template>
        </el-table-column>

        <!-- 操作列 -->
        <el-table-column label="操作" width="100" align="center">
          <template #default="{ row }">
            <el-button
              :type="row.is_locked ? 'success' : 'danger'"
              size="small"
              @click="row.is_locked ? handleUnban(row) : handleBan(row)"
            >
              {{ row.is_locked ? "解封" : "封禁" }}
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup>
import { ref, onMounted } from "vue";
import { Search } from "@element-plus/icons-vue";
import { ElMessage, ElMessageBox } from "element-plus";
import DoctorAPI from "@/api/DoctorAPI";

// 状态变量
const loading = ref(false);
const searchQuery = ref("");
const doctorList = ref([]);
const filteredDoctorList = ref([]);

/**
 * 获取医生列表数据
 * 包含错误处理和加载状态管理
 */
const getDoctorList = async () => {
  loading.value = true;
  try {
    const response = await DoctorAPI.getDoctors();
    if (response.success) {
      // 处理医生列表数据，添加封禁状态标记
      doctorList.value = response.doctors.map((doctor) => ({
        ...doctor,
        is_locked:
          doctor.locked_until && new Date(doctor.locked_until) > new Date(),
      }));
      handleSearch(); // 更新过滤后的列表
    } else {
      ElMessage.error(response.message || "获取医生列表失败");
    }
  } catch (error) {
    console.error("获取医生列表失败:", error);
    ElMessage.error("获取医生列表失败，请检查网络连接");
  } finally {
    loading.value = false;
  }
};

/**
 * 处理搜索逻辑
 * 支持医生姓名、工号和科室的模糊搜索
 */
const handleSearch = () => {
  if (!searchQuery.value) {
    filteredDoctorList.value = doctorList.value;
    return;
  }

  const query = searchQuery.value.toLowerCase();
  filteredDoctorList.value = doctorList.value.filter(
    (doctor) =>
      doctor.name.toLowerCase().includes(query) ||
      doctor.doctor_id.toLowerCase().includes(query) ||
      doctor.department.toLowerCase().includes(query)
  );
};

/**
 * 处理医生封禁操作
 * @param {Object} doctor - 医生信息对象
 */
const handleBan = (doctor) => {
  ElMessageBox.confirm(
    `确定要封禁医生 ${doctor.name} 吗？\n封禁后该医生将在180天内无法登录系统。`,
    "警告",
    {
      confirmButtonText: "确定封禁",
      cancelButtonText: "取消",
      type: "warning",
      confirmButtonClass: "el-button--danger",
    }
  )
    .then(async () => {
      try {
        const response = await DoctorAPI.banDoctor(doctor.doctor_id);
        if (response.success) {
          ElMessage.success(`医生 ${doctor.name} 已被封禁`);
          getDoctorList(); // 刷新列表
        } else {
          throw new Error(response.message || "封禁操作失败");
        }
      } catch (error) {
        console.error("封禁医生失败:", error);
        ElMessage.error(error.message || "封禁失败，请稍后重试");
      }
    })
    .catch(() => {
      ElMessage.info("已取消封禁操作");
    });
};

/**
 * 处理解除医生封禁操作
 * @param {Object} doctor - 医生信息对象
 */
const handleUnban = (doctor) => {
  ElMessageBox.confirm(`确定要解除医生 ${doctor.name} 的封禁状态吗？`, "提示", {
    confirmButtonText: "确定解封",
    cancelButtonText: "取消",
    type: "info",
  })
    .then(async () => {
      try {
        const response = await DoctorAPI.unbanDoctor(doctor.doctor_id);
        if (response.success) {
          ElMessage.success(`已解除医生 ${doctor.name} 的封禁状态`);
          getDoctorList(); // 刷新列表
        } else {
          throw new Error(response.message || "解除封禁操作失败");
        }
      } catch (error) {
        console.error("解除医生封禁失败:", error);
        ElMessage.error(error.message || "解除封禁失败，请稍后重试");
      }
    })
    .catch(() => {
      ElMessage.info("已取消解封操作");
    });
};

// 组件挂载时获取医生列表
onMounted(() => {
  getDoctorList();
});
</script>

<style scoped>
/* 页面布局样式 */
.doctor-management {
  padding: 20px;
  background-color: #f5f7fa;
  min-height: calc(100vh - 60px);
}

/* 医生列表卡片样式 */
.doctor-list {
  max-width: 1200px;
  margin: 0 auto;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

/* 卡片头部样式 */
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 15px 20px;
  border-bottom: 1px solid #ebeef5;
}

.card-header span {
  font-size: 18px;
  font-weight: 600;
  color: #303133;
}

/* 搜索栏样式 */
.search-bar {
  margin: 20px 0;
  display: flex;
  justify-content: flex-start;
}

/* 输入框样式 */
:deep(.el-input) {
  width: 300px;
}

:deep(.el-input__inner) {
  border-radius: 20px;
}

/* 表格样式优化 */
:deep(.el-table) {
  border-radius: 8px;
  overflow: hidden;
}

:deep(.el-table th) {
  background-color: #f5f7fa;
  color: #606266;
  font-weight: 600;
}

:deep(.el-table__row) {
  transition: all 0.3s ease;
}

:deep(.el-table__row:hover) {
  background-color: #f5f7fa;
}

/* 标签样式 */
:deep(.el-tag) {
  border-radius: 12px;
  padding: 0 12px;
}

/* 按钮样式 */
:deep(.el-button--small) {
  padding: 6px 16px;
  border-radius: 15px;
}
</style> 