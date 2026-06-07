#MCU发出的数
Flag mcuflag

#can重启次数
uint8 restart_success_flag

bool mcu_error_flag
bool pps_restart_flag

int32 period_cnt_mcu

#心跳包
HeartBag heartbag

int32 tx_can_cnt
int32 rx_can_cnt

#麦轮
Motor wheel1
Motor wheel2
Motor wheel3
Motor wheel4

#腿
Motor leg1
Motor leg2
Motor leg3
Motor leg4

#大象鼻子
Motor pitch
Motor telescoping
Motor damiao

char ps_num1
char ps_num2