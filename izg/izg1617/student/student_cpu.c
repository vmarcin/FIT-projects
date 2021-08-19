/*!
 * @file 
 * @brief This file contains implementation of cpu side for phong shading.
 *
 * @author Tomáš Milet, imilet@fit.vutbr.cz
 *
 */

#include<assert.h>
#include<math.h>

#include"student/student_cpu.h"
#include"student/student_pipeline.h"
#include"student/linearAlgebra.h"
#include"student/uniforms.h"
#include"student/gpu.h"
#include"student/camera.h"
#include"student/vertexPuller.h"
#include"student/buffer.h"
#include"student/bunny.h"
#include"student/student_shader.h"
#include"student/mouseCamera.h"
#include"student/swapBuffers.h"

#define VERTICES_OF_BUNNY 6276
///This variable contains projection matrix.
extern Mat4 projectionMatrix;
///This variable contains view matrix.
extern Mat4 viewMatrix;
///This variable contains camera position in world-space.
extern Vec3 cameraPosition;

/**
 * @brief This structure contains all global variables for this method.
 */
struct PhongVariables{
  ///This variable contains GPU handle.
  GPU gpu;
  ///This variable contains light poistion in world-space.
  Vec3 lightPosition;
	//This variable contains puller id
	VertexPullerID puller;
	//This variable contains program id
	ProgramID prg;
	//This variable contains vertex buffer object
	BufferID vbo;
	//This variable contains element buffer object
	BufferID ebo;
}phong;///<instance of all global variables for triangle example.

/// \addtogroup cpu_side Úkoly v cpu části
/// @{

void phong_onInit(int32_t width,int32_t height){
  //create gpu
  phong.gpu = cpu_createGPU();
  //set viewport size
  cpu_setViewportSize(phong.gpu,(size_t)width,(size_t)height);
  //init matrices
  cpu_initMatrices(width,height);
  //init lightPosition
  init_Vec3(&phong.lightPosition,100.f,100.f,100.f);

  /// \todo Doprogramujte inicializační funkci.
  /// Zde byste měli vytvořit buffery na GPU, nahrát data do bufferů, vytvořit vertex puller a správně jej nakonfigurovat, vytvořit program, připojit k němu shadery a nastavit interpolace.
  /// Také byste zde měli zarezervovat unifromní proměnné pro matice, pozici kamery, světla a další vaše proměnné.
  /// Do bufferů nahrajte vrcholy králička (pozice, normály) a indexy na vrcholy ze souboru bunny.h.
  /// Shader program by měl odkazovat na funkce/shadery v souboru student_shader.h.
  /// V konfiguraci vertex pulleru nastavte dvě čtecí hlavy.
  /// Jednu pro pozice vrcholů a druhou pro normály vrcholů.
  /// Nultý vertex/fragment atribut by měl obsahovat pozici vertexu.
  /// První vertex/fragment atribut by měl obsahovat normálu vertexu.
  /// Budete využívat minimálně 4 uniformní proměnné
  /// Uniformní proměnná pro view matici by měla být pojmenována "viewMatrix".
  /// Uniformní proměnná pro projekční matici by měla být pojmenována "projectionMatrix".
  /// Uniformní proměnná pro pozici kamery by se měla jmenovat "cameraPosition".
  /// Uniformní proměnná pro pozici světla by se měla jmenovat "lightPosition".
  /// Je důležité udržet pozice atributů a názvy uniformních proměnných z důvodu akceptačních testů.
  /// Interpolace vertex atributů do fragment atributů je také potřeba nastavit.
  /// Oba vertex atributy nastavte na \link SMOOTH\endlink interpolaci - perspektivně korektní interpolace.<br>
  /// <b>Seznam funkcí, které jistě využijete:</b>
  ///  - cpu_reserveUniform()
  ///  - cpu_createProgram()
  ///  - cpu_attachVertexShader()
  ///  - cpu_attachFragmentShader()
  ///  - cpu_setAttributeInterpolation()
  ///  - cpu_createBuffers()
  ///  - cpu_bufferData()
  ///  - cpu_createVertexPullers()
  ///  - cpu_setVertexPullerHead()
  ///  - cpu_enableVertexPullerHead()
  ///  - cpu_setIndexing()

	//reserve uniforms
	cpu_reserveUniform(phong.gpu,"projectionMatrix",UNIFORM_MAT4);
	cpu_reserveUniform(phong.gpu,"viewMatrix"			 ,UNIFORM_MAT4);
	cpu_reserveUniform(phong.gpu,"cameraPosition"	 ,UNIFORM_VEC3);
	cpu_reserveUniform(phong.gpu,"lightPosition"	 ,UNIFORM_VEC3);
	
	//create program		
	phong.prg = cpu_createProgram(phong.gpu);
	cpu_attachVertexShader	(phong.gpu,phong.prg,phong_vertexShader);
	cpu_attachFragmentShader(phong.gpu,phong.prg,phong_fragmentShader);
	
	//set interpolation
	cpu_setAttributeInterpolation(phong.gpu,phong.prg,0,ATTRIB_VEC3,SMOOTH);
	cpu_setAttributeInterpolation(phong.gpu,phong.prg,1,ATTRIB_VEC3,SMOOTH);
	
	//create buffers
	BufferID ids[2];
	cpu_createBuffers(phong.gpu,2,ids);
	phong.vbo = ids[0];
	phong.ebo = ids[1];
	
	//upload data to buffers
	cpu_bufferData(phong.gpu,phong.vbo,sizeof(bunnyVertices),bunnyVertices);
	cpu_bufferData(phong.gpu,phong.ebo,sizeof(bunnyIndices),bunnyIndices);
	
	//create VertexPuller
	cpu_createVertexPullers(phong.gpu,1,&(phong.puller));
	
	//set position attribute
	cpu_setVertexPullerHead(phong.gpu,phong.puller,0,phong.vbo,0,sizeof(float)*6);
	//set normal attribute
	cpu_setVertexPullerHead(phong.gpu,phong.puller,1,phong.vbo,sizeof(float)*3,sizeof(float)*6);
	
	//enable vertexPuller heads
	cpu_enableVertexPullerHead(phong.gpu,phong.puller,0);
	cpu_enableVertexPullerHead(phong.gpu,phong.puller,1);
	
	//set indexing
	cpu_setIndexing(phong.gpu,phong.puller,phong.ebo,sizeof(unsigned));
}

/// @}

void phong_onExit(){
  cpu_destroyGPU(phong.gpu);
}

/// \addtogroup cpu_side
/// @{

void phong_onDraw(SDL_Surface*surface){
  assert(surface != NULL);

  //clear depth buffer
  cpu_clearDepth(phong.gpu,+INFINITY);
  Vec4 color;
  init_Vec4(&color,.1f,.1f,.1f,1.f);
  //clear color buffer
  cpu_clearColor(phong.gpu,&color);

  /// \todo Doprogramujte kreslící funkci.
  /// Zde byste měli aktivovat shader program, aktivovat vertex puller, nahrát data do uniformních proměnných a 
  /// vykreslit trojúhelníky pomocí funkce cpu_drawTriangles.
  /// Data pro uniformní proměnné naleznete v externích globálních proměnnénych viewMatrix, projectionMatrix, cameraPosition a globální proměnné phong.lightPosition.<br>
  /// <b>Seznam funkcí, které jistě využijete:</b>
  ///  - cpu_useProgram()
  ///  - cpu_bindVertexPuller()
  ///  - cpu_uniformMatrix4fv()
  ///  - cpu_uniform3f()
  ///  - cpu_drawTriangles()
  ///  - getUniformLocation()

	//activate program
	cpu_useProgram(phong.gpu,phong.prg);	
	
	//activate VertexPuller
	cpu_bindVertexPuller(phong.gpu,phong.puller);
	
	//upload uniform matrices
	UniformLocation const viewMatrixUniform = getUniformLocation(phong.gpu,"viewMatrix");
	cpu_uniformMatrix4fv(phong.gpu,viewMatrixUniform,(float*)&viewMatrix);

	UniformLocation const projectionMatrixUniform = getUniformLocation(phong.gpu,"projectionMatrix");
	cpu_uniformMatrix4fv(phong.gpu,projectionMatrixUniform,(float*)&projectionMatrix);
	
	//upload uniform data	
	cpu_uniform3f(
		phong.gpu																			,
		getUniformLocation(phong.gpu,"cameraPosition"),
		cameraPosition.data[0]												,
		cameraPosition.data[1]												,
		cameraPosition.data[2]											 	);
  cpu_uniform3f(
		phong.gpu																			,
		getUniformLocation(phong.gpu,"lightPosition" ),
		phong.lightPosition.data[0]										,
		phong.lightPosition.data[1]										,
		phong.lightPosition.data[2]										);
	
	//draw
	cpu_drawTriangles(phong.gpu,VERTICES_OF_BUNNY);

  //copy image from gpu to SDL surface
  cpu_swapBuffers(surface,phong.gpu);
}

/// @}
