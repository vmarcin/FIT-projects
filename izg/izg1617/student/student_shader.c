/*!
 * @file 
 * @brief This file contains implemenation of phong vertex and fragment shader.
 *
 * @author Tomáš Milet, imilet@fit.vutbr.cz
 */
#include<math.h>
#include<assert.h>

#include"student/student_shader.h"
#include"student/gpu.h"
#include"student/uniforms.h"

/// \addtogroup shader_side Úkoly v shaderech
/// @{

void phong_vertexShader(
    GPUVertexShaderOutput     *const output,
    GPUVertexShaderInput const*const input ,
    GPU                        const gpu   ){
  /// \todo Naimplementujte vertex shader, který transformuje vstupní vrcholy do clip-space.<br>
  /// <b>Vstupy:</b><br>
  /// Vstupní vrchol by měl v nultém atributu obsahovat pozici vrcholu ve world-space (vec3) a v prvním
  /// atributu obsahovat normálu vrcholu ve world-space (vec3).<br>
  /// <b>Výstupy:</b><br>
  /// Výstupní vrchol by měl v nultém atributu obsahovat pozici vrcholu (vec3) ve world-space a v prvním
  /// atributu obsahovat normálu vrcholu ve world-space (vec3).
  /// Výstupní vrchol obsahuje pozici a normálu vrcholu proto, že chceme počítat osvětlení ve world-space ve fragment shaderu.<br>
  /// <b>Uniformy:</b><br>
  /// Vertex shader by měl pro transformaci využít uniformní proměnné obsahující view a projekční matici.
  /// View matici čtěte z uniformní proměnné "viewMatrix" a projekční matici čtěte z uniformní proměnné "projectionMatrix".
  /// Zachovejte jména uniformních proměnných a pozice vstupních a výstupních atributů.
  /// Pokud tak neučiníte, akceptační testy selžou.<br>
  /// <br>
  /// Využijte vektorové a maticové funkce.
  /// Nepředávajte si data do shaderu pomocí globálních proměnných.
  /// Pro získání dat atributů použijte příslušné funkce vs_interpret* definované v souboru program.h.
  /// Pro získání dat uniformních proměnných použijte příslušné funkce shader_interpretUniform* definované v souboru program.h.
  /// Vrchol v clip-space by měl být zapsán do proměnné gl_Position ve výstupní struktuře.<br>
  /// <b>Seznam funkcí, které jistě použijete</b>:
  ///  - gpu_getUniformsHandle()
  ///  - getUniformLocation()
  ///  - shader_interpretUniformAsMat4()
  ///  - vs_interpretInputVertexAttributeAsVec3()
  ///  - vs_interpretOutputVertexAttributeAsVec3()
	
	output->gpu = gpu;
	//get handle to all uniforms
	Uniforms const uniformsHandle = gpu_getUniformsHandle(gpu);
	
	//get uniform location of view matrix
	UniformLocation const viewMatrixLocation = getUniformLocation(gpu,"viewMatrix");
	
	//get uniform location of projection matrix
	UniformLocation const projectionMatrixLocation = getUniformLocation(gpu,"projectionMatrix");

	//get pointer to projection matrix
	Mat4 const*const proj = shader_interpretUniformAsMat4(uniformsHandle,projectionMatrixLocation);
	
	//get pointer to view matrix
	Mat4 const*const view = shader_interpretUniformAsMat4(uniformsHandle,viewMatrixLocation);

	//get pointer to position attribute	(input)
	Vec3 const*const fragPosition_in = vs_interpretInputVertexAttributeAsVec3(gpu,input,0);
	//get pointer to normal attribute (input)
	Vec3 const*const normal_in = vs_interpretInputVertexAttributeAsVec3(gpu,input,1);

	//project vertex position into clip-space
	Mat4 mvp;
	multiply_Mat4_Mat4(&mvp,proj,view);
	Vec4 pos4;
	copy_Vec3Float_To_Vec4(&pos4,fragPosition_in,1.f);
	multiply_Mat4_Vec4(&output->gl_Position,&mvp,&pos4);

	//get pointer to position attribute (output)
	Vec3 *const fragPosition_out = vs_interpretOutputVertexAttributeAsVec3(gpu,output,0); 
	//get pointer to normal attribute (output)
	Vec3 *const normal_out			 = vs_interpretOutputVertexAttributeAsVec3(gpu,output,1);

	//copy position from input to output attribute
	init_Vec3(fragPosition_out,fragPosition_in->data[0],fragPosition_in->data[1],fragPosition_in->data[2]);
	//copy normal from input to output attribute
	init_Vec3(normal_out,normal_in->data[0],normal_in->data[1],normal_in->data[2]);
}

void phong_fragmentShader(
    GPUFragmentShaderOutput     *const output,
    GPUFragmentShaderInput const*const input ,
    GPU                          const gpu   ){
  /// \todo Naimplementujte fragment shader, který počítá phongův osvětlovací model s phongovým stínováním.<br>
  /// <b>Vstup:</b><br>
  /// Vstupní fragment by měl v nultém fragment atributu obsahovat interpolovanou pozici ve world-space a v prvním
  /// fragment atributu obsahovat interpolovanou normálu ve world-space.<br>
  /// <b>Výstup:</b><br> 
  /// Barvu zapište do proměnné color ve výstupní struktuře.<br>
  /// <b>Uniformy:</b><br>
  /// Pozici kamery přečtěte z uniformní proměnné "cameraPosition" a pozici světla přečtěte z uniformní proměnné "lightPosition".
  /// Zachovejte jména uniformních proměnný.
  /// Pokud tak neučiníte, akceptační testy selžou.<br>
  /// <br>
  /// Dejte si pozor na velikost normálového vektoru, při lineární interpolaci v rasterizaci může dojít ke zkrácení.
  /// Zapište barvu do proměnné color ve výstupní struktuře.
  /// Shininess faktor nastavte na 40.f
  /// Difuzní barvu materiálu nastavte na čistou zelenou.
  /// Spekulární barvu materiálu nastavte na čistou bílou.
  /// Barvu světla nastavte na bílou.
  /// Nepoužívejte ambientní světlo.<br>
  /// <b>Seznam funkcí, které jistě využijete</b>:
  ///  - shader_interpretUniformAsVec3()
  ///  - fs_interpretInputAttributeAsVec3()

	//get handle to all uniforms
	Uniforms const uniformsHandle = gpu_getUniformsHandle(gpu);

	//get uniform location of camera position	
	UniformLocation const cameraPositionLocation = getUniformLocation(gpu,"cameraPosition");
	//get uniform location of light position
	UniformLocation const lightPositionLocation  = getUniformLocation(gpu,"lightPosition");

	//get pointer to camera position
	Vec3 const*const camera = shader_interpretUniformAsVec3(uniformsHandle,cameraPositionLocation);
	//get pointer to light position
	Vec3 const*const light  = shader_interpretUniformAsVec3(uniformsHandle,lightPositionLocation);
	
	//get pointer to zeroth (fragment position) input fragment attribute
	Vec3 const *fragPosition = fs_interpretInputAttributeAsVec3(gpu,input,0);
	//get pointer to first (normal position) input fragment attribute
	Vec3 const *normal  	 	 = fs_interpretInputAttributeAsVec3(gpu,input,1);

	//init secular light	
	Vec3 specularLight	  = {.data[0] = 1.f, .data[1] = 1.f, .data[2] = 1.f};
	//init diffuse light
	Vec3 diffuseLight 		= {.data[0] = 0.f, .data[1] = 1.f, .data[2] = 0.f};

	Vec3 norm;		
	normalize_Vec3(&norm,normal); //normalize normal
			
	Vec3 lightDirection;			
	Vec3 viewDirection;			
	Vec3 reflectDirection;	
				
	Vec3 diffuse; 					
	Vec3 specular;					
	Vec3 phong;		
	
	//compute diffuse part 	
	sub_Vec3			(&lightDirection,light,fragPosition);
	normalize_Vec3(&lightDirection,&lightDirection);
	
	float diff = dot_Vec3(&norm,&lightDirection);
	if(diff < 0.f)
		diff = 0.f;
	if(diff > 1.f)
		diff = 1.f;
	multiply_Vec3_Float(&diffuse,&diffuseLight,diff);

	//compute specular part
	sub_Vec3			(&viewDirection,camera,fragPosition);
	normalize_Vec3(&viewDirection,&viewDirection);
	
	multiply_Vec3_Float(&lightDirection,&lightDirection,-1.0f);	
	reflect(&reflectDirection,&lightDirection,&norm);
	
	float spec = dot_Vec3(&viewDirection,&reflectDirection);
	if(spec < 0.f)
		spec = 0.f;
	if(spec > 1.f)
		spec = 1.f;
	spec = (float)pow(spec,40.f);	
	multiply_Vec3_Float(&specular,&specularLight,spec);
	
	//compute phong reflection
	add_Vec3(&phong,&specular,&diffuse);
	
	//write color to output fragment
	copy_Vec3Float_To_Vec4(&output->color,&phong,1.f);
}

/// @}
