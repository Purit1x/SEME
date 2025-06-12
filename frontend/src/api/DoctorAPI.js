import request from '@/utils/request'
const BASE_URL = 'http://localhost:5000'// 后端API的基地址

//预测函数
const predict=async(data)=>{
    try{
        const response = await request.post(`${BASE_URL}/api/predictions/predict`,data)
        return response
    }catch(error){
        console.error('Failed to predict',error)
        throw error
    }
}

// 获取医生列表
const getDoctors = async () => {
    try {
        const response = await request.get(`${BASE_URL}/api/auth/doctors`)
        return response
    } catch (error) {
        console.error('获取医生列表失败', error)
        throw error
    }
}

// 封禁医生
const banDoctor = async (doctorId) => {
    try {
        const response = await request.post(`${BASE_URL}/api/auth/doctor/${doctorId}/ban`)
        return response
    } catch (error) {
        console.error('封禁医生失败', error)
        throw error
    }
}

// 解除封禁
const unbanDoctor = async (doctorId) => {
    try {
        const response = await request.post(`${BASE_URL}/api/auth/doctor/${doctorId}/unban`)
        return response
    } catch (error) {
        console.error('解除封禁失败', error)
        throw error
    }
}

export default {
    predict,
    getDoctors,
    banDoctor,
    unbanDoctor
}




