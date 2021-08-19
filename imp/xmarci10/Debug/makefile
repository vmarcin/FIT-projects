################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include utilities/subdir.mk
-include startup/subdir.mk
-include source/subdir.mk
-include drivers/subdir.mk
-include board/subdir.mk
-include CMSIS/subdir.mk
-include subdir.mk
-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
endif

-include ../makefile.defs

# Add inputs and outputs from these tool invocations to the build variables 

# All Target
all: twrk60d100m_demo_apps_hello_world.axf

# Tool invocations
twrk60d100m_demo_apps_hello_world.axf: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: MCU Linker'
	arm-none-eabi-gcc -nostdlib -Xlinker -Map="twrk60d100m_demo_apps_hello_world.map" -Xlinker --gc-sections -Xlinker -print-memory-usage -Xlinker --sort-section=alignment -mcpu=cortex-m4 -mthumb -T twrk60d100m_demo_apps_hello_world_Debug.ld -o "twrk60d100m_demo_apps_hello_world.axf" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '
	$(MAKE) --no-print-directory post-build

# Other Targets
clean:
	-$(RM) $(EXECUTABLES)$(OBJS)$(C_DEPS) twrk60d100m_demo_apps_hello_world.axf
	-@echo ' '

post-build:
	-@echo 'Performing post-build steps'
	-arm-none-eabi-size "twrk60d100m_demo_apps_hello_world.axf"; # arm-none-eabi-objcopy -v -O binary "twrk60d100m_demo_apps_hello_world.axf" "twrk60d100m_demo_apps_hello_world.bin" ; # checksum -p MK60DN512xxx10 -d "twrk60d100m_demo_apps_hello_world.bin";
	-@echo ' '

.PHONY: all clean dependents post-build

-include ../makefile.targets
