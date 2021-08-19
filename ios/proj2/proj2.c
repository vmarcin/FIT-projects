/*
 * proj2.c
 * Riesenie IOS-proj2, 27.4.2017
 * Autor: Vladimír Marcin FIT 1-BIT (xmarci10)
 * Prelozene: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609
 */
#include "proj2.h" 

int *children; 					//pocet deti v centre
int children_id;

int *adults; 						//pocet dospelych v centre
int adults_id;

int *waiting; 					//pocet deti cakajucich na vstup do centra 
int waiting_id;

int *leaving; 					//pocet dospelych cakajucich na odchod z centra
int leaving_id;

int *counter; 					//pocitadlo akcii
int counter_id;

int *adults_coming; 		//pocet dospelych ktory este maju prist
int adults_coming_id;

int *process_counter; 	//pocitadlo procesov
int process_counter_id;

sem_t *sem_mutex;				//mutex semafor
sem_t *sem_childQueue;	//fronta pre deti ktore chcu vstupit
sem_t *sem_adultQueue;	//fronta pre dospelych ktory chcu oddist
sem_t *sem_end;					//fronta pre vsetky procesy (aby sa ukoncili naraz)

FILE *fp=NULL;

int main(int argc, char *argv[])
{
	srand(time(NULL));
	pid_t children_generator;		//pid procesu ktory generuje deti
	pid_t adults_generator;			//pid procesu ktory generuje dospelych
	
	arguments arg = test_argument(argc,argv);	

	fp = fopen("proj2.out","w");
	if(fp == NULL)
	{
		fprintf(stderr,"CHYBA: Nepodarilo sa otvorit subor!\n");
		return 2;
	}
	
	//vytvorenie zdrojov (semafory a zdielana pamet)	
	create_semaphores();
	if((create_shm(arg)) == false)
	{
		fprintf(stderr,"CHYBA: Nepodarilo sa vytvorit zdielane premenne!\n");
		free_semaphores();
		fclose(fp);
		return 2;
	}
	setbuf(fp,NULL);
	setbuf(stderr,NULL);
	//setbuf(stdout,NULL);

	if((children_generator = fork()) == 0)
	{	
		// proces generujuci children procesy
		pid_t *children_p = malloc(sizeof(pid_t)*arg.children);
		for(int i = 0; i < arg.children; i++)
		{
			usleep(random_number(arg.cgt)*1000);
			if((children_p[i] == (children_generator = fork())) == 0)
			{
				child(arg.cwt,i+1);
				exit(0);
			}
			//fork sa nepodaril
			else if(children_generator < 0)
			{
				fprintf(stderr,"CHYBA: Chyba pri vytvarani child procesov!\n");
				for(int j = 0; j < i; j++)
					kill(children_p[j],SIGTERM);
		
				free(children_p);
				free_shm();
				free_semaphores();
				fclose(fp);
				exit(2);
			}
		}
		//proces generujuci children prosecy pocka na ich ukoncenie
		for(int i = 0; i < arg.children; i++)
			waitpid(children_p[i], NULL, 0);
		free(children_p);
		exit(0);
	}
	else if(children_generator < 0)
	{
		fprintf(stderr,"CHYBA: Chyba pri vytvarani procesov!\n");
		free_shm();
		free_semaphores();
		fclose(fp);
		exit(2);
	}
	else if((adults_generator = fork()) == 0)
	{
		//proces generujuci adult procesy
		pid_t *adults_p = malloc(sizeof(pid_t)*arg.adults);
		for(int i = 0; i < arg.adults; i++)
		{
			usleep(random_number(arg.agt)*1000);
			if((adults_p[i] = (adults_generator = fork())) == 0)
			{
				adult(arg.awt,i+1);
				exit(0);
			}
			else if (adults_generator < 0)
			{
				fprintf(stderr,"CHYBA: Chyba pri vytvarani adults procesov!\n");
				for(int j = 0; j < i; j++)
					kill(adults_p[j],SIGTERM);

				free(adults_p);
				free_shm();
				free_semaphores();
				fclose(fp);
				exit(2);
			}
		}
		//proces generujuci adult procesy caka na ich ukoncenie
		for(int i = 0; i < arg.adults; i++)
			waitpid(adults_p[i],NULL, 0);
		free(adults_p);
		exit(0);
	}
	else if(adults_generator < 0)
	{
		fprintf(stderr,"CHYBA: Chyba pri vytvarani procesov!\n");
		free_shm();
		free_semaphores();
		fclose(fp);
		kill(children_generator,SIGTERM);
		exit(2);
	}
	//hlavny proces caka na ukoncenie procesov, ktore
	//generuju children a adult procesy
	waitpid(children_generator, NULL, 0);
	waitpid(adults_generator, NULL, 0);

	//uvolnime zdroje
	free_shm();
	free_semaphores();
	fclose(fp);
	return 0;
}

void adult(int awt, int id)
{
	//adult proces zacal
	sem_wait(sem_mutex);
		fprintf(fp,"%-8d: A %-4d: started\n",(*counter)++,id);
		fflush(fp);
	sem_post(sem_mutex);	
	//adult proces vstupil do centra
	sem_wait(sem_mutex);
		fprintf(fp,"%-8d: A %-4d: enter\n",(*counter)++,id);
		fflush(fp);
		*adults+=1;				
		*adults_coming-=1;	//pocet prichadzajucich procesov adult sa znizi	
		//ak nejake deti cakaju na vstup do centra adult im posle signal
		//ze mozu vstupit
		if(*waiting)
		{
			//jeden adult sa moze "starat" max. o 3child procesy
			int n = (*waiting < 3) ? *waiting : 3;
			//povolime vstup max. 3procesom ak je pocet cakajucich child
			//menej ako tri posleme iba tolko signalov kolko child je cakajucich
			for(int i = 0; i < n; i++)
					sem_post(sem_childQueue);
			*waiting-=n;
			*children+=n;
		}
		sem_post(sem_mutex);
	
	//KRITICKA SEKCIA
	//simulujeme cinnost procesu adult pocas nahodnej doby
	usleep(random_number(awt)*1000);

	sem_wait(sem_mutex);
		//po prebudeni sa proces adult pokusa oddist
		fprintf(fp,"%-8d: A %-4d: trying to leave\n",(*counter)++,id);
		fflush(fp);
		//ak jeho odchodom nebude porusena podmienka centra oddide
		if(*children <= 3*(*adults-1))
		{
			fprintf(fp,"%-8d: A %-4d: leave\n",(*counter)++,id);
			fflush(fp);
			*adults-=1;
			sem_post(sem_mutex);
		}
		else
		{
			//ak by jeho odchodom bola porusena podmienka zaradi
			//sa do fornty cakajucich na odchod
			fprintf(fp,"%-8d: A %-4d: waiting : %d : %d\n",(*counter)++, id, *adults, *children);
			fflush(fp);
			*leaving+=1;
			sem_post(sem_mutex);
			sem_wait(sem_adultQueue);
			
			//po odchode potrebneho poctu child proces adul proces
			//dostane signal a odpusta centrum
			sem_wait(sem_mutex);
				fprintf(fp,"%-8d: A %-4d: leave\n",(*counter)++,id);
				fflush(fp);
			sem_post(sem_mutex);
		}
	
	//proces opustil centrum cize pocet procesov
	//znizime aby sme vedeli identifikovat ukoncenie
	sem_wait(sem_mutex);
		*process_counter-=1;
	sem_post(sem_mutex);
	
	//ak je pocet prichadzajucich rodicov rovny nule
	//povolime vstup vsetkym cakajucim procesom child
	if(*adults_coming == 0 && *adults == 0)
	{
		for(int i = 0; i < *waiting; i++)
			sem_post(sem_childQueue);
	}

	//ak je pocet procesov rovny nule
	//znamena ze vsetky procesy opustili centrum
	//tak vsetky procesy sa ukoncia sucastne
	if(*process_counter == 0)
		sem_post(sem_end);
	sem_wait(sem_end);
	sem_post(sem_end);

	sem_wait(sem_mutex);
		fprintf(fp,"%-8d: A %-4d: finished\n",(*counter)++,id);
		fflush(fp);
	sem_post(sem_mutex);
}

void child(int cwt, int id)
{
	//child proces zacal
	sem_wait(sem_mutex);
		fprintf(fp,"%-8d: C %-4d: started\n",(*counter)++,id);
		fflush(fp);
	sem_post(sem_mutex);
	
	sem_wait(sem_mutex);
		//ak je splnena podmienka centra alebo ak je pocet prichadzajucich adult
		//procesov rovny 0 a zaroven sa v centre nenachadza rodic proces child 
		//moze vstupit do centra
		if((*children < (3 * (*adults))) || (*adults_coming == 0 && *adults == 0))
		{
			fprintf(fp,"%-8d: C %-4d: enter\n",(*counter)++,id);
			fflush(fp);
			*children+=1;
			sem_post(sem_mutex);
		}
		//ak by vstupom dietata bola porusena podmienka centra child proces sa zaradi
		//do fronty cakajucich procesov a caka na vstup noveho adult procesu
		else
		{
			fprintf(fp,"%-8d: C %-4d: waiting : %d : %d\n",(*counter)++, id, *adults, *children);
			fflush(fp);
			*waiting+=1;
			sem_post(sem_mutex);
			sem_wait(sem_childQueue);
			
			//po prichode adult do centra adult posle signal max. 3 cakaucim detom
			//ktore vstupia do centra
			sem_wait(sem_mutex);
				fprintf(fp,"%-8d: C %-4d: enter\n",(*counter)++,id);
				fflush(fp);
			sem_post(sem_mutex);
		}
	
	//KRITICKA SEKCIA
	//proces child simuluje svoju cinnost pocas nohodnej doby
	usleep(random_number(cwt)*1000);	
	
	sem_wait(sem_mutex);
		//po prebudeni sa child proces pokusa oddist a kedze jeho odchod
		//nieje nicim podmieneny opusta centrum
		fprintf(fp,"%-8d: C %-4d: trying to leave\n",(*counter)++,id);
		fflush(fp);
		fprintf(fp,"%-8d: C %-4d: leave\n",(*counter)++,id);
		fflush(fp);
		*children-=1;
		*process_counter-=1;
		//ak sa vo fronte rodicov cakajucich na odchod z centra nachadza
		//nejaky proces (leaving) a centrum opustilo dostatocne mnozstvo
		//deti posle sa signal procesu adult ze moze opustit centrum
		if(*leaving && *children <= 3 * (*adults-1))
		{
			*leaving-=1;
			*adults-=1;
			sem_post(sem_adultQueue);
		}
	sem_post(sem_mutex);

	//ak je pocet procesov rovny nule
	//znamena ze vsetky procesy opustili centrum
	//tak vsetky procesy sa ukoncia sucastne
	if(*process_counter == 0)
		sem_post(sem_end);
	sem_wait(sem_end);
	sem_post(sem_end);

	sem_wait(sem_mutex);
		fprintf(fp,"%-8d: C %-4d: finished\n",(*counter)++,id);
		fflush(fp);
	sem_post(sem_mutex);
}

void create_semaphores()
{
	if((sem_mutex = sem_open(semMUTEX, O_CREAT | O_EXCL | O_RDWR, 0666, 1)) == SEM_FAILED)
		goto close;	
	if((sem_childQueue = sem_open(semCHILD_QUEUE, O_CREAT | O_EXCL | O_RDWR, 0666, 0)) == SEM_FAILED)
		goto close1;
	if((sem_adultQueue = sem_open(semADULT_QUEUE, O_CREAT | O_EXCL | O_RDWR, 0666, 0)) == SEM_FAILED)
		goto close2;
	if((sem_end = sem_open(semEND, O_CREAT | O_EXCL | O_RDWR, 0666, 0)) == SEM_FAILED)
		goto close3;
	
	goto end;
	
	close3:
		sem_close(sem_adultQueue);
		sem_unlink(semADULT_QUEUE);
	close2:
		sem_close(sem_childQueue);
		sem_unlink(semCHILD_QUEUE);
	close1:
		sem_close(sem_mutex);
		sem_unlink(semMUTEX);
	close:
		fclose(fp);
		fprintf(stderr,"CHYBA: Nepodarilo sa vytvorit semafory!\n");
		exit(2);
	end:
		return;
}

void free_semaphores()
{
	//ak nastane chyba pri uvolnovani semaforov 
	//premenna error bude mat hodnotu true
	bool error = false;
	if(			(sem_close(sem_mutex)) == -1 ||
		 (sem_close(sem_childQueue)) == -1 ||
		 (sem_close(sem_adultQueue)) == -1 ||
		 				(sem_close(sem_end)) == -1)
		error = true;
	if(			 (sem_unlink(semMUTEX)) == -1 ||
		 (sem_unlink(semCHILD_QUEUE)) == -1 ||
	   (sem_unlink(semADULT_QUEUE)) == -1 ||
						 (sem_unlink(semEND)) == -1) 
		error = true;

	if(error)
	{
		fprintf(stderr,"VAROVANIE: Nepodarilo sa zatvorit vsetky semafory!\n");
		fclose(fp);
		exit(2);
	}
}

bool create_shm(arguments arg)
{
	if((children_id = shm_open(shmCHILDREN, O_CREAT | O_EXCL | O_RDWR, S_IRUSR | S_IWUSR)) == -1)
		goto jump1;
	ftruncate(children_id,shmSIZE);

	if((adults_id = shm_open(shmADULTS, O_CREAT | O_EXCL | O_RDWR, S_IRUSR | S_IWUSR)) == -1)
		goto jump2;
	ftruncate(adults_id,shmSIZE);

	if((waiting_id = shm_open(shmWAITING, O_CREAT | O_EXCL | O_RDWR, S_IRUSR | S_IWUSR)) == -1)
		goto jump3;
	ftruncate(waiting_id,shmSIZE);

	if((leaving_id = shm_open(shmLEAVING, O_CREAT | O_EXCL | O_RDWR, S_IRUSR | S_IWUSR)) == -1)
		goto jump4;
	ftruncate(leaving_id,shmSIZE);

	if((counter_id = shm_open(shmCOUNTER, O_CREAT | O_EXCL | O_RDWR, S_IRUSR | S_IWUSR)) == -1)
		goto jump5;
	ftruncate(counter_id,shmSIZE);

	if((adults_coming_id = shm_open(shmADULTS_COMING, O_CREAT | O_EXCL | O_RDWR, S_IRUSR | S_IWUSR)) == -1)
		goto jump6;
	ftruncate(adults_coming_id,shmSIZE);

	if((process_counter_id = shm_open(shmPROCESS_COUNTER, O_CREAT | O_EXCL | O_RDWR, S_IRUSR | S_IWUSR)) == -1)
		goto jump7;
	ftruncate(process_counter_id,shmSIZE);

	children = (int*)mmap(NULL, shmSIZE, PROT_READ | PROT_WRITE, MAP_SHARED, children_id, 0);
	if(children == MAP_FAILED)
		goto jump8;
	close(children_id);
	*children = 0;

	adults = (int*)mmap(NULL, shmSIZE, PROT_READ | PROT_WRITE, MAP_SHARED, adults_id, 0);
	if(adults == MAP_FAILED)
		goto jump9;
	close(adults_id);
	*adults = 0;

	waiting = (int*)mmap(NULL, shmSIZE, PROT_READ | PROT_WRITE, MAP_SHARED, waiting_id, 0);
	if(waiting == MAP_FAILED)
		goto jump10;
	close(waiting_id);
	*waiting = 0;

	leaving = (int*)mmap(NULL, shmSIZE, PROT_READ | PROT_WRITE, MAP_SHARED, leaving_id, 0);
	if(leaving == MAP_FAILED)
		goto jump11;
	close(leaving_id);
	*leaving = 0;
	
	counter = (int*)mmap(NULL, shmSIZE, PROT_READ | PROT_WRITE, MAP_SHARED, counter_id, 0);
	if(counter == MAP_FAILED)
		goto jump12;
	close(counter_id);
	*counter = 1;

	adults_coming = (int*)mmap(NULL, shmSIZE, PROT_READ | PROT_WRITE, MAP_SHARED, adults_coming_id, 0);
	if(adults_coming == MAP_FAILED)
		goto jump13;
	close(adults_coming_id);
	*adults_coming = arg.adults;

	process_counter = (int*)mmap(NULL, shmSIZE, PROT_READ | PROT_WRITE, MAP_SHARED, process_counter_id, 0);
	if(process_counter == MAP_FAILED)
		goto jump14;
	close(process_counter_id);
	*process_counter = arg.adults + arg.children;
	
	goto finish;

	jump14: munmap(adults_coming,shmSIZE);
	jump13: munmap(counter,shmSIZE);
	jump12: munmap(leaving,shmSIZE);
	jump11: munmap(waiting,shmSIZE);
	jump10: munmap(adults,shmSIZE);
	jump9:  munmap(children,shmSIZE);
	jump8:	shm_unlink(shmPROCESS_COUNTER);
	jump7:  shm_unlink(shmADULTS_COMING);
	jump6:  shm_unlink(shmCOUNTER);
	jump5:  shm_unlink(shmLEAVING);
	jump4:  shm_unlink(shmWAITING);
	jump3:  shm_unlink(shmADULTS);
	jump2:  shm_unlink(shmCHILDREN);
	jump1:  return false; 

	finish:
		return true;	
}

void free_shm()
{
	//ak nastane chyba pri uvolnovani zdielanej pamete
	//premenna error bude mat hodnotu true
	bool error = false;

	if( 	 (munmap(children,shmSIZE)) == -1 ||
			  	 (munmap(adults,shmSIZE)) == -1 ||
		   		(munmap(waiting,shmSIZE)) == -1 ||
			  	(munmap(leaving,shmSIZE)) == -1 ||
			  	(munmap(counter,shmSIZE)) == -1 ||
		(munmap(adults_coming,shmSIZE)) == -1 ||
	(munmap(process_counter,shmSIZE)) == -1)
		error = true;

	if( 	 (shm_unlink(shmCHILDREN)) == -1 || 
			  	 (shm_unlink(shmADULTS)) == -1 ||
			 		(shm_unlink(shmWAITING)) == -1 ||
			 		(shm_unlink(shmLEAVING)) == -1 ||
			 		(shm_unlink(shmCOUNTER)) == -1 ||
		(shm_unlink(shmADULTS_COMING)) == -1 ||
	(shm_unlink(shmPROCESS_COUNTER)) == -1)
		error = true;
	//ak nastala chyba vypise sa varovanie a ukonci sa program
	if(error)
	{
		fprintf(stderr,"VAROVANIE: Nepodarilo sa zavriet vsetky zdielane premenne!\n");
		fclose(fp);
		free_semaphores();
		exit(2);	
	}
}

unsigned random_number(int max)
{
	if(max == 0)
		return 0;
	return rand() % (max+1);
}

void usage()
{
	fprintf(stderr,"USAGE: ./proj2 A C AGT CGT AWT CWT\n\
\tA je pocet procesov adult; A > 0\n\
\tC je pocet procesov child; C > 0\n\
\tAGT je maximalna doba (ms) generovania procesu adult; AGT >= 0 && AGT < 5001\n\
\tCGT je maximalne doba (ms) generovania prosesu child; CGT >= 0 && CGT < 5001\n\
\tAWT je maximalna doba (ms) behu procesu adult; AWT >= 0 && AWT < 5001\n\
\tCWT je maximalna doba (ms) behu procesu child; CWT >= 0 && CWT < 5001\n");
}
arguments test_argument(int argc, char *argv[])
{
	arguments test;
	int tmp;		
	char *ptr;
	if(argc != 7)
	{
		fprintf(stderr,"CHYBA: Nespravny pocet argumentov!\n");
		usage();
		exit(1);
	}

	tmp = (int)strtol(argv[1],&ptr,10);
	if(tmp <= 0 || *ptr != 0)
	{
		fprintf(stderr,"CHYBA: Nespravna hodnota argumentu 'A'!\n");
		exit(1);
	}
	test.adults = tmp;
	
	tmp = (int)strtol(argv[2],&ptr,10);
	if(tmp <= 0 || *ptr != 0)
	{
		fprintf(stderr,"CHYBA: Nespravna hodnota argumentu 'C'!\n");
		usage();
		exit(1);
	}
	test.children = tmp;
	
	tmp = (int)strtol(argv[3],&ptr,10);
	if(tmp < 0 || tmp >= 5001 || *ptr != 0)
	{
		fprintf(stderr,"CHYBA: Nespravna hodnota argumentu 'AGT'!\n");
		usage();
		exit(1);
	}
	test.agt = tmp;
	
	tmp = (int)strtol(argv[4],&ptr,10);
	if(tmp < 0 || tmp >= 5001 || *ptr != 0)
	{
		fprintf(stderr,"CHYBA: Nespravna hodnota argumentu 'CGT'!\n");
		usage();
		exit(1);
	}
	test.cgt = tmp;
	
	tmp = (int)strtol(argv[5],&ptr,10);
	if(tmp < 0 || tmp >= 5001 || *ptr != 0)
	{
		fprintf(stderr,"CHYBA: Nespravna hodnota argumentu 'AWT'!\n");
		usage();
		exit(1);
	}
	test.awt = tmp;
	
	tmp = (int)strtol(argv[6],&ptr,10);
	if(tmp < 0 || tmp >= 5001 || *ptr != 0)
	{
		fprintf(stderr,"CHYBA: Nespravna hodnota argumentu 'CWT'!\n");
		usage();
		exit(1);
	}
	test.cwt = tmp;

	return test;
}
