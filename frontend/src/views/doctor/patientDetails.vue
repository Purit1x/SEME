<template>
  <div class="patient-details">
    <el-card class="patient-info">
      <template #header>
        <div class="card-header">
          <span>患者详情信息</span>
          <div class="header-buttons">
            <el-button type="primary" @click="handleMRIManage">MRI序列管理</el-button>
            <el-button @click="$router.back()">返回</el-button>
          </div>
        </div>
      </template>

      <div class="info-content" v-if="patientInfo">
        <div class="patient-photo">
          <img :src="getFullImageUrl(patientInfo.photo_url)" alt="患者照片" v-if="patientInfo.photo_url">
          <el-avatar v-else icon="el-icon-user" :size="120"></el-avatar>
        </div>

        <el-descriptions :column="2" border>
          <el-descriptions-item label="姓名">{{ patientInfo.name }}</el-descriptions-item>
          <el-descriptions-item label="性别">{{ patientInfo.sex }}</el-descriptions-item>
          <el-descriptions-item label="年龄">{{ patientInfo.age }}</el-descriptions-item>
          <el-descriptions-item label="身份证号">{{ patientInfo.id_number }}</el-descriptions-item>
        </el-descriptions>
      </div>
      <div v-else class="no-data">
        <el-empty description="未找到患者信息"></el-empty>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'

const route = useRoute()
const router = useRouter()
const patientInfo = ref(null)

// 获取完整的图片URL
const getFullImageUrl = (url) => {
  if (!url) return ''

  // 打印环境变量和URL信息
  console.log('环境变量信息：', {
    VITE_API_URL: import.meta.env.VITE_API_URL,
    MODE: import.meta.env.MODE
  })

  // 如果已经是完整URL，直接返回
  if (url.startsWith('http://') || url.startsWith('https://')) {
    console.log('使用完整URL:', url)
    return url
  }

  // 否则拼接基础URL
  const fullUrl = `${import.meta.env.VITE_API_URL}${url}`
  console.log('拼接后的URL:', fullUrl)
  return fullUrl
}

// 跳转到MRI序列管理页面
const handleMRIManage = () => {
  router.push({
    name: 'patientMRI',
    params: { patientId: patientInfo.value.id }
  })
}

onMounted(() => {
  console.log('当前路由信息：', {
    query: route.query,
    path: route.path
  })

  const patientInfoData = route.query.patientInfo
  console.log('接收到的患者信息数据：', patientInfoData)

  if (patientInfoData) {
    try {
      const decodedData = decodeURIComponent(patientInfoData)
      console.log('解码后的数据：', decodedData)
      patientInfo.value = JSON.parse(decodedData)
      console.log('患者详细信息：', {
        姓名: patientInfo.value.name,
        年龄: patientInfo.value.age,
        性别: patientInfo.value.sex,
        身份证号: patientInfo.value.id_number,
        照片URL: getFullImageUrl(patientInfo.value.photo_url)
      })
    } catch (error) {
      console.error('解析患者信息失败:', error)
      console.error('错误详情:', {
        message: error.message,
        stack: error.stack
      })
    }
  } else {
    console.warn('未接收到患者信息')
  }
})
</script>

<style scoped>
.patient-details {
  padding: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header-buttons {
  display: flex;
  gap: 10px;
}

.patient-info {
  max-width: 800px;
  margin: 0 auto;
}

.patient-photo {
  text-align: center;
  margin-bottom: 20px;
}

.patient-photo img {
  width: 120px;
  height: 120px;
  border-radius: 60px;
  object-fit: cover;
  border: 1px solid #eee;
}

.info-content {
  margin-top: 20px;
}

.no-data {
  padding: 40px 0;
}
</style>