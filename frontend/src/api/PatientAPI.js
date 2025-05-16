import axios from 'axios'
import utils from './utils'
import request from '@/utils/request'
const BASE_URL = 'http://localhost:5000'// 后端API的基地址

const getPatientPhoto=async(patientId)=>{
    try{
        const response = await request.get(`${BASE_URL}/api/patients/${patientId}/photo`)
        return response
    }catch(error){
        console.error('Failed to get patient photo',error)
        throw error
    }
}

