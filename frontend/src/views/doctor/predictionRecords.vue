<template>
    <div class="prediction-records">
        <el-card class="records-list">
            <template #header>
                <div class="card-header">
                    <div class="header-left">
                        <el-icon class="header-icon"><Document /></el-icon>
                        <span>预测记录列表</span>
                    </div>
                    <div class="header-right">
                        <span class="record-count">共 {{ filteredRecords.length }} 条记录</span>
                    </div>
                </div>
            </template>

            <!-- 搜索区域 -->
            <div class="search-area">
                <el-input
                    v-model="searchQuery"
                    placeholder="输入患者姓名搜索"
                    clearable
                    @clear="handleSearch"
                    @input="handleSearch"
                    class="search-input"
                >
                    <template #prefix>
                        <el-icon><Search /></el-icon>
                    </template>
                </el-input>
            </div>

            <!-- 记录列表 -->
            <el-table 
                :data="filteredRecords" 
                style="width: 100%" 
                v-loading="loading"
                :header-cell-style="{ background: '#f5f7fa', color: '#606266' }"
                :row-class-name="tableRowClassName"
                @row-click="handleRowClick"
            >
                <el-table-column prop="patientName" label="患者姓名" min-width="120">
                    <template #default="scope">
                        <div class="patient-name">
                            <el-icon><User /></el-icon>
                            <span>{{ scope.row.patientName }}</span>
                        </div>
                    </template>
                </el-table-column>
                <el-table-column prop="sequenceName" label="序列名称" min-width="120">
                    <template #default="scope">
                        <el-tag size="small" effect="plain">{{ scope.row.sequenceName }}</el-tag>
                    </template>
                </el-table-column>
                <el-table-column prop="predTime" label="预测时间" min-width="180">
                    <template #default="scope">
                        <div class="pred-time">
                            <el-icon><Timer /></el-icon>
                            <span>{{ formatDate(scope.row.predTime) }}</span>
                        </div>
                    </template>
                </el-table-column>
                <el-table-column label="操作" width="120" fixed="right">
                    <template #default="scope">
                        <el-button 
                            type="primary" 
                            link 
                            @click.stop="showDetails(scope.row)"
                            class="detail-button"
                        >
                            <el-icon><View /></el-icon>
                            查看详情
                        </el-button>
                    </template>
                </el-table-column>
            </el-table>
        </el-card>

        <!-- 详情对话框 -->
        <el-dialog
            v-model="detailsVisible"
            title="预测记录详情"
            width="70%"
            :before-close="handleCloseDialog"
            class="details-dialog"
        >
            <div class="record-details" v-if="selectedRecord" v-loading="imageLoading">
                <div class="details-section">
                    <h3>
                        <el-icon><InfoFilled /></el-icon>
                        基本信息
                        <el-button 
                            type="primary" 
                            link 
                            class="show-all-button"
                            @click="showAllRecords"
                        >
                            <el-icon><List /></el-icon>
                            查看所有记录
                        </el-button>
                    </h3>
                    <el-descriptions :column="2" border>
                        <el-descriptions-item label="患者姓名">
                            <el-tag size="small">{{ selectedRecord.patientName }}</el-tag>
                        </el-descriptions-item>
                        <el-descriptions-item label="序列名称">
                            <el-tag size="small" type="success">{{ selectedRecord.sequenceName }}</el-tag>
                        </el-descriptions-item>
                        <el-descriptions-item label="预测时间">
                            <el-tag size="small" type="info">{{ formatDate(selectedRecord.predTime) }}</el-tag>
                        </el-descriptions-item>
                    </el-descriptions>
                </div>

                <div class="images-section">
                    <h3>
                        <el-icon><Picture /></el-icon>
                        预测图像
                    </h3>
                    <div class="image-grid">
                        <div class="image-item">
                            <div class="image-header">
                                <el-icon><Camera /></el-icon>
                                <h4>原始RGB图像</h4>
                            </div>
                            <div class="image-container" @click="previewImage(getImageUrl(selectedRecord.processed_rgb_path))">
                                <img :src="getImageUrl(selectedRecord.processed_rgb_path)" alt="原始RGB图像" />
                                <div class="image-overlay">
                                    <el-icon><ZoomIn /></el-icon>
                                </div>
                            </div>
                        </div>
                        <div class="image-item">
                            <div class="image-header">
                                <el-icon><Star /></el-icon>
                                <h4>预测结果</h4>
                            </div>
                            <div class="image-container" @click="previewImage(getImageUrl(selectedRecord.result_path))">
                                <img :src="getImageUrl(selectedRecord.result_path)" alt="预测结果" />
                                <div class="image-overlay">
                                    <el-icon><ZoomIn /></el-icon>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </el-dialog>

        <!-- 侧拉栏 -->
        <el-drawer
            v-model="drawerVisible"
            title="所有预测记录"
            direction="rtl"
            size="500px"
            :destroy-on-close="false"
        >
            <div class="drawer-content">
                <div class="drawer-header">
                    <el-input
                        v-model="drawerSearchQuery"
                        placeholder="搜索记录..."
                        clearable
                        class="drawer-search"
                    >
                        <template #prefix>
                            <el-icon><Search /></el-icon>
                        </template>
                    </el-input>
                    <div class="selection-controls">
                        <el-button type="primary" link @click="selectAll" :disabled="selectedRecords.length === filteredDrawerRecords.length">
                            全选
                        </el-button>
                        <el-button type="danger" link @click="clearSelection" :disabled="selectedRecords.length === 0">
                            清空选择
                        </el-button>
                        <span class="selection-count">已选择 {{ selectedRecords.length }} 项</span>
                    </div>
                </div>
                
                <el-scrollbar height="calc(100vh - 260px)">
                    <div class="record-list">
                        <div 
                            v-for="record in showSelectedOnly ? selectedRecords : filteredDrawerRecords" 
                            :key="record.pred_id"
                            class="record-item"
                            :class="{ 'selected': isSelected(record) }"
                            @click="toggleRecordSelection(record)"
                        >
                            <div class="record-item-header">
                                <el-tag size="small">{{ record.patientName }}</el-tag>
                                <el-tag size="small" type="success">{{ record.sequenceName }}</el-tag>
                            </div>
                            <div class="record-item-time">
                                <el-icon><Timer /></el-icon>
                                <span>{{ formatDate(record.predTime) }}</span>
                            </div>
                            <div class="record-item-preview">
                                <div class="preview-image">
                                    <img :src="getImageUrl(record.processed_rgb_path)" alt="布针图" />
                                </div>
                                <div class="preview-image">
                                    <img :src="getImageUrl(record.result_path)" alt="预测结果" />
                                </div>
                            </div>
                        </div>
                    </div>
                </el-scrollbar>

                <!-- 底部确认按钮 -->
                <div class="drawer-footer">
                    <el-button @click="showSelectedOnly = !showSelectedOnly">
                        {{ showSelectedOnly ? '显示全部' : '仅显示已选' }}
                    </el-button>
                    <el-button type="primary" @click="confirmSelection" :disabled="selectedRecords.length === 0">
                        确认选择
                    </el-button>
                </div>
            </div>
        </el-drawer>

        <!-- 图片预览 -->
        <el-image-viewer
            v-if="showViewer"
            :url-list="[previewUrl]"
            @close="showViewer = false"
        />
    </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { 
    Search, Document, User, Timer, View, InfoFilled, 
    Picture, Camera, Star, ZoomIn, List 
} from '@element-plus/icons-vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'

const loading = ref(false)
const imageLoading = ref(false)
const records = ref([])
const searchQuery = ref('')
const detailsVisible = ref(false)
const selectedRecord = ref(null)
const showViewer = ref(false)
const previewUrl = ref('')

// 侧拉栏相关
const drawerVisible = ref(false)
const drawerSearchQuery = ref('')
const selectedRecords = ref([])
const showSelectedOnly = ref(false)

// 显示所有记录
const showAllRecords = () => {
    drawerVisible.value = true
}

// 过滤侧拉栏记录
const filteredDrawerRecords = computed(() => {
    if (!drawerSearchQuery.value) return records.value
    
    const query = drawerSearchQuery.value.toLowerCase()
    return records.value.filter(record => 
        record.patientName.toLowerCase().includes(query) ||
        record.sequenceName.toLowerCase().includes(query)
    )
})

// 选择记录
const toggleRecordSelection = (record) => {
    const index = selectedRecords.value.findIndex(r => r.pred_id === record.pred_id && r.predTime === record.predTime)
    if (index === -1) {
        selectedRecords.value.push({ ...record })
    } else {
        selectedRecords.value.splice(index, 1)
    }
}

// 检查记录是否被选中
const isSelected = (record) => {
    return selectedRecords.value.some(r => r.pred_id === record.pred_id && r.predTime === record.predTime)
}

// 全选
const selectAll = () => {
    selectedRecords.value = filteredDrawerRecords.value.map(record => ({ ...record }))
}

// 清空选择
const clearSelection = () => {
    selectedRecords.value = []
}

// 获取预测记录列表
const getRecords = async () => {
    loading.value = true
    try {
        const token = localStorage.getItem('access_token')
        const response = await axios.get('http://localhost:5000/api/predictions/records', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        })
        
        if (response.data.success) {
            records.value = response.data.records
            console.log('获取到的预测记录:', records.value)
        } else {
            throw new Error(response.data.message || '获取预测记录失败')
        }
    } catch (error) {
        console.error('获取预测记录失败:', error)
        ElMessage.error(error.response?.data?.message || '获取预测记录失败')
    } finally {
        loading.value = false
    }
}

// 格式化日期
const formatDate = (dateString) => {
    if (!dateString) return ''
    const date = new Date(dateString)
    return date.toLocaleString()
}

// 处理搜索
const handleSearch = () => {
    // 搜索逻辑在计算属性中实现
}

// 过滤后的记录列表
const filteredRecords = computed(() => {
    if (!searchQuery.value) return records.value
    
    const query = searchQuery.value.toLowerCase()
    return records.value.filter(record => 
        record.patientName.toLowerCase().includes(query)
    )
})

// 显示详情
const showDetails = async (record) => {
    imageLoading.value = true
    selectedRecord.value = record
    detailsVisible.value = true
    imageLoading.value = false
}

// 关闭对话框
const handleCloseDialog = () => {
    selectedRecord.value = null
    detailsVisible.value = false
}

// 获取图片URL
const getImageUrl = (path) => {
    if (!path) return ''
    const normalizedPath = path.replace(/\\/g, '/')
    return `http://localhost:5000/${normalizedPath}`
}

// 表格行样式
const tableRowClassName = ({ row, rowIndex }) => {
    return 'table-row'
}

// 处理行点击
const handleRowClick = (row) => {
    showDetails(row)
}

// 预览图片
const previewImage = (url) => {
    previewUrl.value = url
    showViewer.value = true
}

// 确认选择
const confirmSelection = () => {
    showSelectedOnly.value = true
    ElMessage.success(`已选择 ${selectedRecords.value.length} 条记录`)
}

onMounted(() => {
    getRecords()
})
</script>

<style scoped>
.prediction-records {
    padding: 20px;
    background-color: #f5f7fa;
    min-height: calc(100vh - 84px);
}

.records-list {
    background-color: #fff;
    border-radius: 8px;
    box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.header-left {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 16px;
    font-weight: bold;
}

.header-icon {
    font-size: 20px;
    color: #409eff;
}

.record-count {
    color: #909399;
    font-size: 14px;
}

.search-area {
    margin-bottom: 20px;
}

.search-input {
    max-width: 300px;
}

.patient-name, .pred-time {
    display: flex;
    align-items: center;
    gap: 6px;
}

.detail-button {
    display: flex;
    align-items: center;
    gap: 4px;
}

.table-row {
    cursor: pointer;
    transition: background-color 0.3s;
}

.table-row:hover {
    background-color: #f5f7fa;
}

.record-details {
    padding: 20px;
}

.details-section {
    margin-bottom: 30px;
}

.details-section h3 {
    margin-bottom: 15px;
    color: #333;
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 16px;

    .show-all-button {
        margin-left: auto;
        font-size: 14px;
    }
}

.image-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
    margin-top: 20px;
}

.image-item {
    display: flex;
    flex-direction: column;
    background-color: #f5f7fa;
    border-radius: 8px;
    padding: 15px;
}

.image-header {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 10px;
}

.image-header h4 {
    margin: 0;
    color: #606266;
    font-size: 14px;
}

.image-container {
    position: relative;
    cursor: pointer;
    border-radius: 4px;
    overflow: hidden;
}

.image-container img {
    width: 100%;
    height: auto;
    display: block;
    transition: transform 0.3s;
}

.image-overlay {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    opacity: 0;
    transition: opacity 0.3s;
}

.image-overlay .el-icon {
    font-size: 24px;
    color: #fff;
}

.image-container:hover img {
    transform: scale(1.05);
}

.image-container:hover .image-overlay {
    opacity: 1;
}

:deep(.el-dialog) {
    border-radius: 8px;
}

:deep(.el-dialog__header) {
    margin-right: 0;
    border-bottom: 1px solid #dcdfe6;
    padding: 20px;
}

:deep(.el-dialog__body) {
    padding: 0;
}

:deep(.el-descriptions__cell) {
    padding: 12px 20px;
}

/* 侧拉栏样式 */
.drawer-content {
    padding: 16px 16px 72px 16px;
    height: 100%;
    position: relative;
}

.drawer-header {
    margin-bottom: 20px;
}

.drawer-search {
    margin-bottom: 16px;
}

.selection-controls {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0;
    border-bottom: 1px solid #ebeef5;
}

.selection-count {
    margin-left: auto;
    color: #606266;
    font-size: 14px;
}

.record-list {
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 16px 0;
}

.record-item {
    background: white;
    border-radius: 8px;
    padding: 16px;
    cursor: pointer;
    transition: all 0.3s ease;
    border: 2px solid transparent;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);

    &:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }

    &.selected {
        border-color: #409eff;
        background-color: #ecf5ff;
    }
}

.record-item-header {
    display: flex;
    gap: 8px;
    margin-bottom: 8px;
}

.record-item-time {
    display: flex;
    align-items: center;
    gap: 4px;
    color: #606266;
    font-size: 14px;
    margin-bottom: 12px;
}

.record-item-preview {
    display: flex;
    gap: 12px;
    margin-top: 12px;
}

.preview-image {
    flex: 1;
    aspect-ratio: 1;
    border-radius: 4px;
    overflow: hidden;
    background-color: #f5f7fa;

    img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
}

:deep(.el-drawer__header) {
    margin-bottom: 0;
    padding: 16px 20px;
    border-bottom: 1px solid #ebeef5;
}

:deep(.el-drawer__body) {
    padding: 0;
    overflow: hidden;
}

:deep(.el-scrollbar__view) {
    padding-right: 16px;
}

.drawer-footer {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 16px;
    background-color: white;
    border-top: 1px solid #ebeef5;
    display: flex;
    justify-content: flex-end;
    gap: 12px;
    box-shadow: 0 -2px 8px rgba(0, 0, 0, 0.05);
}
</style> 