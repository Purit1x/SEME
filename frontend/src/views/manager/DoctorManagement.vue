<template>
    <div class="doctor-management">
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
            <el-table :data="filteredDoctorList" style="width: 100%" v-loading="loading">
                <el-table-column prop="doctor_id" label="工号" width="120" />
                <el-table-column prop="name" label="姓名" width="120" />
                <el-table-column prop="department" label="科室" width="120" />
                <el-table-column prop="email" label="邮箱" />
                <el-table-column label="状态" width="120">
                    <template #default="scope">
                        <el-tag
                            :type="scope.row.is_locked ? 'danger' : 'success'"
                            size="small"
                        >
                            {{ scope.row.is_locked ? '已封禁' : '正常' }}
                        </el-tag>
                    </template>
                </el-table-column>
                <el-table-column label="操作" width="120">
                    <template #default="scope">
                        <el-button
                            v-if="!scope.row.is_locked"
                            size="small"
                            type="danger"
                            @click="handleBan(scope.row)"
                        >
                            封禁
                        </el-button>
                        <el-button
                            v-else
                            size="small"
                            type="success"
                            @click="handleUnban(scope.row)"
                        >
                            解封
                        </el-button>
                    </template>
                </el-table-column>
            </el-table>
        </el-card>
    </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { Search } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import DoctorAPI from '@/api/DoctorAPI'

const loading = ref(false)
const searchQuery = ref('')
const doctorList = ref([])
const filteredDoctorList = ref([])

// 获取医生列表
const getDoctorList = async () => {
    loading.value = true
    try {
        const response = await DoctorAPI.getDoctors()
        if (response.success) {
            // 添加封禁状态
            doctorList.value = response.doctors.map(doctor => ({
                ...doctor,
                is_locked: doctor.locked_until && new Date(doctor.locked_until) > new Date()
            }))
            handleSearch()
        } else {
            ElMessage.error(response.message || '获取医生列表失败')
        }
    } catch (error) {
        console.error('获取医生列表失败:', error)
        ElMessage.error('获取医生列表失败')
    } finally {
        loading.value = false
    }
}

// 搜索处理
const handleSearch = () => {
    if (!searchQuery.value) {
        filteredDoctorList.value = doctorList.value
        return
    }

    const query = searchQuery.value.toLowerCase()
    filteredDoctorList.value = doctorList.value.filter(doctor => 
        doctor.name.toLowerCase().includes(query) ||
        doctor.doctor_id.toLowerCase().includes(query) ||
        doctor.department.toLowerCase().includes(query)
    )
}

// 封禁医生
const handleBan = (row) => {
    ElMessageBox.confirm(
        `确定要封禁医生 ${row.name} 吗？封禁后该医生将在180天内无法登录系统。`,
        '警告',
        {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning',
        }
    ).then(async () => {
        try {
            const response = await DoctorAPI.banDoctor(row.doctor_id)
            if (response.success) {
                ElMessage.success('医生已被封禁')
                getDoctorList() // 刷新列表
            } else {
                ElMessage.error(response.message || '封禁失败')
            }
        } catch (error) {
            console.error('封禁医生失败:', error)
            ElMessage.error('封禁失败')
        }
    }).catch(() => {
        ElMessage.info('已取消封禁')
    })
}

// 解除封禁
const handleUnban = (row) => {
    ElMessageBox.confirm(
        `确定要解除医生 ${row.name} 的封禁状态吗？`,
        '提示',
        {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning',
        }
    ).then(async () => {
        try {
            const response = await DoctorAPI.unbanDoctor(row.doctor_id)
            if (response.success) {
                ElMessage.success('已解除医生封禁')
                getDoctorList() // 刷新列表
            } else {
                ElMessage.error(response.message || '解除封禁失败')
            }
        } catch (error) {
            console.error('解除医生封禁失败:', error)
            ElMessage.error('解除封禁失败')
        }
    }).catch(() => {
        ElMessage.info('已取消操作')
    })
}

onMounted(() => {
    getDoctorList()
})
</script>

<style scoped>
.doctor-management {
    padding: 20px;
}

.doctor-list {
    max-width: 1200px;
    margin: 0 auto;
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.search-bar {
    margin-bottom: 20px;
}

:deep(.el-input) {
    max-width: 300px;
}
</style> 