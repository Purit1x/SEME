<template>
    <div class="sequence-detail">
        <el-card class="detail-card">
            <template #header>
                <div class="card-header">
                    <div class="header-left">
                        <span class="title">序列详情</span>
                        <span class="sequence-info" v-if="sequenceInfo">
                            - {{ sequenceInfo.name }}
                        </span>
                    </div>
                    <div class="header-right">
                        <el-button @click="$router.back()">返回</el-button>
                    </div>
                </div>
            </template>

            <div v-loading="loading">
                <template v-if="sequenceInfo">
                    <div class="sequence-meta">
                        <el-descriptions :column="2" border>
                            <el-descriptions-item label="序列名称">{{ sequenceInfo.name }}</el-descriptions-item>
                            <el-descriptions-item label="创建时间">
                                {{ new Date(sequenceInfo.created_at).toLocaleString() }}
                            </el-descriptions-item>
                            <el-descriptions-item label="患者姓名">{{ sequenceInfo.patient_name }}</el-descriptions-item>
                            <el-descriptions-item label="图像数量">{{ sequenceInfo.items?.length || 0 }}
                                组</el-descriptions-item>
                        </el-descriptions>
                    </div>

                    <div class="image-list">
                        <el-table :data="sequenceInfo.items" style="width: 100%" v-if="sequenceInfo.items?.length">
                            <el-table-column label="序号" type="index" width="80" />
                            <el-table-column label="RGB图像" width="300">
                                <template #default="scope">
                                    <el-image :src="scope.row.rgb.path" :preview-src-list="[scope.row.rgb.path]"
                                        fit="contain" class="preview-image">
                                        <template #error>
                                            <div class="image-error">
                                                <el-icon><picture-filled /></el-icon>
                                                <span>加载失败</span>
                                            </div>
                                        </template>
                                    </el-image>
                                </template>
                            </el-table-column>
                            <el-table-column label="深度图像" width="300">
                                <template #default="scope">
                                    <el-image :src="scope.row.depth.path" :preview-src-list="[scope.row.depth.path]"
                                        fit="contain" class="preview-image">
                                        <template #error>
                                            <div class="image-error">
                                                <el-icon><picture-filled /></el-icon>
                                                <span>加载失败</span>
                                            </div>
                                        </template>
                                    </el-image>
                                </template>
                            </el-table-column>
                            <el-table-column label="上传时间">
                                <template #default="scope">
                                    {{ new Date(scope.row.rgb.uploaded_at).toLocaleString() }}
                                </template>
                            </el-table-column>
                        </el-table>
                        <el-empty v-else description="暂无图像" />
                    </div>
                </template>
                <el-empty v-else description="未找到序列信息" />
            </div>
        </el-card>
    </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { PictureFilled } from '@element-plus/icons-vue'
import axios from 'axios'

const route = useRoute()
const router = useRouter()
const loading = ref(false)
const sequenceInfo = ref(null)

// 获取完整的图片URL
const getFullImageUrl = (url) => {
    if (!url) return ''
    return url
}

// 获取序列详情
const fetchSequenceDetail = async () => {
    try {
        loading.value = true
        const { patientId, sequenceId } = route.params
        console.log('正在获取序列详情:', {
            patientId,
            sequenceId,
            url: `${import.meta.env.VITE_API_URL}/api/mri/patients/${patientId}/sequences/${sequenceId}`
        })

        const response = await axios.get(
            `${import.meta.env.VITE_API_URL}/api/mri/patients/${patientId}/sequences/${sequenceId}`,
            {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('access_token')}`
                }
            }
        )

        console.log('API响应:', response.data)

        if (response.data.success) {
            sequenceInfo.value = response.data.sequence
            // 处理图片URL
            if (sequenceInfo.value.items) {
                sequenceInfo.value.items = sequenceInfo.value.items.map(item => ({
                    ...item,
                    rgb: {
                        ...item.rgb,
                        path: getFullImageUrl(item.rgb.path)
                    },
                    depth: {
                        ...item.depth,
                        path: getFullImageUrl(item.depth.path)
                    }
                }))
            }
        } else {
            ElMessage.error(response.data.message || '获取序列详情失败')
        }
    } catch (error) {
        console.error('获取序列详情失败:', error)
        console.error('错误详情:', {
            message: error.message,
            config: error.config,
            response: error.response
        })
        ElMessage.error('获取序列详情失败，请稍后重试')
    } finally {
        loading.value = false
    }
}

onMounted(() => {
    fetchSequenceDetail()
})
</script>

<style scoped>
.sequence-detail {
    padding: 20px;
}

.detail-card {
    max-width: 1200px;
    margin: 0 auto;
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.header-left {
    display: flex;
    align-items: center;
    gap: 10px;
}

.title {
    font-size: 18px;
    font-weight: bold;
}

.sequence-info {
    color: #666;
}

.header-right {
    display: flex;
    gap: 10px;
}

.sequence-meta {
    margin-bottom: 20px;
}

.image-list {
    margin-top: 20px;
}

.preview-image {
    width: 200px;
    height: 150px;
    border-radius: 4px;
    border: 1px solid #eee;
}

.image-error {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: #909399;
    font-size: 14px;
    gap: 8px;
    height: 100%;
}

:deep(.el-descriptions) {
    margin-bottom: 20px;
}

:deep(.el-table) {
    margin-top: 20px;
}
</style>