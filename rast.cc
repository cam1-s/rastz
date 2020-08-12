#define _USE_MATH_DEFINES
#include <cmath>

#include <thread>
#include <vector>
#include <string>
#include <chrono>
#include <iostream>

#include <glm/glm.hpp>
#include <glm/ext.hpp>

// hack to remove "undefined reference to WinMain" on MINGW
#define SDL_main_h_
#include <SDL2/SDL.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#define M_PI_2 1.57079632679489661923
#endif

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include "inline.hh"

typedef float float_type;
typedef glm::tvec3<float_type> vec3;
typedef glm::tvec2<float_type> vec2;
typedef glm::tvec4<float_type> vec4;
typedef glm::tmat3x3<float_type> mat3;
typedef glm::tmat4x4<float_type> mat4;

constexpr float_type _2PI = 2. * M_PI;

float_type const clip_near = .03;
float_type const clip_far = 10000.;

float_type const fov = 70.;

unsigned const width = 1280;
unsigned const height = 720;

vec3 const light_pos(5., 10., 2.);
vec3 const light_col(1., 1., 1.);
vec3 const ambient_col(.1);

std::vector<uint32_t> main_image(width * height);
std::vector<float_type> depth_buffer(width * height);

mat4 mat_projection;
mat4 mat_view;

vec4 viewport; // x = near, y = far, z = width, w = height
std::vector<uint32_t> *bound_framebuffer;
std::vector<float_type> *bound_depthbuffer;

class vertex
{
public:
	vertex() {  }
	vertex(vec3 const &p, vec3 const &n, vec3 const &c, vec2 const &t) : pos(p), nrm(n), col(c), uv(t) {  }
	vertex(vertex const &v) { operator=(v); }

	vertex &operator=(vertex const &v)
	{
		pos = v.pos, nrm = v.nrm, col = v.col, uv = v.uv;
		return *this;
	}

	vec3 pos, nrm, col;
	vec2 uv;
};

class triangle
{
public:
	triangle() {  }
	triangle(vertex const &_0, vertex const &_1, vertex const &_2) : v0(_0), v1(_1), v2(_2) {  }
	triangle(triangle const &t) { operator=(t); } 

	triangle &operator=(triangle const &t) { v0 = t.v0, v1 = t.v1, v2 = t.v2; return *this; }

	vertex v0, v1, v2;
};

class model
{
public:
	model() {  }
	model(std::vector<triangle> const &v, glm::mat4 const &m) : data(v), transform(m) {  }
	model(model const &m) { operator=(m); }

	model &operator=(model const &m)
	{
		transform = m.transform;
		data = m.data;
		return *this;
	}

	glm::mat4 transform;
	std::vector<triangle> data;
};

std::vector<model> scene;

void bind_buffers(std::vector<uint32_t> &framebuffer, std::vector<float_type> &depth)
{
	bound_framebuffer = &framebuffer;
	bound_depthbuffer = &depth;
}

void clear_buffers()
{
	size_t sz = viewport.z * viewport.w;
	#pragma omp for
	for (size_t i = 0; i < sz; i++) (*bound_framebuffer)[i] = 0, (*bound_depthbuffer)[i] = viewport.y;
}

__always_inline vec4 calc_vertex(vertex const &vert, model const &m)
{
	// clip space <- camera space <- world space <- model space
	return mat_projection * mat_view * m.transform * vec4(vert.pos, 1.);
}

__always_inline vec3 calc_fragment(vec3 const &frag_pos, vertex const &vert)
{
	//vec3 ret = frag_pos / vec3(width, height, 1);
	//ret.z = (frag_pos.z - .15f) * 4.f;
	//return (vert.nrm + 1.f) * .5f;

	vec3 nrm = glm::normalize(vert.nrm);

	vec3 light_dir = glm::normalize(light_pos - vert.pos);
	float_type diffuse = glm::max(glm::dot(nrm, light_dir), 0.f);

	vec3 view_dir = glm::normalize(frag_pos - vert.pos);
	vec3 refl_dir = glm::reflect(-light_dir, nrm);

	float_type specular = .5f * glm::pow(glm::max(glm::dot(view_dir, refl_dir), 0.f), 32);

	return (ambient_col + light_col * (diffuse + specular)) * vec3(.6, .6, 1.);
}

__always_inline vec3 clip_to_screen(vec4 const &clip_coord)
{
	// convert clip-space coordinates to NDC coordinates
	vec3 v = vec3(clip_coord) / clip_coord.w;

	// convert NDC coordinates to raster-space coordinates
	v.x = (v.x + 1.) * .5 * viewport.z;
	v.y = (1. - v.y) * .5 * viewport.w;

	// convert NDC z coordinate to real depth
	v.z = (v.z + 1.) * .5 * (viewport.y - viewport.x) + viewport.x;
	return v;
}

__always_inline float_type inside_edge(vec3 const &a, vec3 const &b, vec3 const &x)
{
	return (x.x - a.x) * (b.y - a.y) - (x.y - a.y) * (b.x - a.x);
}

void render_tri(triangle const &_t, model const &_m)
{
	vec3 p0 = clip_to_screen(calc_vertex(_t.v0, _m));
	vec3 p1 = clip_to_screen(calc_vertex(_t.v1, _m));
	vec3 p2 = clip_to_screen(calc_vertex(_t.v2, _m));

	vec3 min = glm::min(glm::min(p0, p1), p2);
	vec3 max = glm::max(glm::max(p0, p1), p2);

	if (min.z > clip_far || min.x > viewport.z || min.y > viewport.w) return;
	if (max.z < clip_near || max.x < 0 || max.y < 0) return;

	min.x = glm::clamp(min.x, 0.f, static_cast<float_type>(viewport.z));
	min.y = glm::clamp(min.y, 0.f, static_cast<float_type>(viewport.w));
	max.x = glm::clamp(max.x, 0.f, static_cast<float_type>(viewport.z));
	max.y = glm::clamp(max.y, 0.f, static_cast<float_type>(viewport.w));

	// perspective correction
	vertex v0(_t.v0.pos / p0.z, _t.v0.nrm / p0.z, _t.v0.col, _t.v0.uv / p0.z);
	vertex v1(_t.v1.pos / p0.z, _t.v1.nrm / p1.z, _t.v1.col, _t.v1.uv / p0.z);
	vertex v2(_t.v2.pos / p0.z, _t.v2.nrm / p2.z, _t.v2.col, _t.v2.uv / p0.z);

	/*vertex v0(_t.v0.pos, _t.v0.nrm, _t.v0.col, _t.v0.uv);
	vertex v1(_t.v1.pos, _t.v1.nrm, _t.v1.col, _t.v1.uv);
	vertex v2(_t.v2.pos, _t.v2.nrm, _t.v2.col, _t.v2.uv);
	*/
	vec3 rz = 1.f / vec3(p0.z, p1.z, p2.z);

	for (int i = min.x; i < max.x; i++) for (int j = min.y; j < max.y; j++)
	{
		vec3 p(i, j, 1);

		vec3 e(inside_edge(p1, p2, p), inside_edge(p2, p0, p), inside_edge(p0, p1, p));
		
		/*if ((glm::abs(p0.x - i) < 1 && glm::abs(p0.y - j) < 1) ||
		    (glm::abs(p1.x - i) < 1 && glm::abs(p1.y - j) < 1) ||
		    (glm::abs(p2.x - i) < 1 && glm::abs(p2.y - j) < 1))*/
		if (e.x > 0.f && e.y > 0.f && e.z > 0.f)
		{
			// barycentric coordinates
			vec3 w = e / inside_edge(p0, p1, p2);
			
			float_type depth = 1.f / glm::dot(rz, w);

			if (depth < clip_near || (*bound_depthbuffer)[((j) * viewport.z) + i] < depth) continue;
			(*bound_depthbuffer)[((j) * viewport.z) + i] = depth;

			// interpolated vertex
			vertex v(w.x * v0.pos + w.y * v1.pos + w.z * v2.pos,
			         w.x * v0.nrm + w.y * v1.nrm + w.z * v2.nrm,
			         w.x * v0.col + w.y * v1.col + w.z * v2.col,
			         w.x * v0.uv  + w.y * v1.uv  + w.z * v2.uv);

			// perspective correction
			v.pos *= depth, v.nrm *= depth, v.col *= depth, v.uv *= depth;

			vec3 fc = calc_fragment(p, v);
			fc = glm::clamp(fc, vec3(0.), vec3(1.));

			uint32_t c = 0xff000000;
			c |= static_cast<uint8_t>(fc.x * 0xff) << 16;
			c |= static_cast<uint8_t>(fc.y * 0xff) << 8;
			c |= static_cast<uint8_t>(fc.z * 0xff);

			(*bound_framebuffer)[((j) * viewport.z) + i] = c;
		}
	}
}

std::vector<triangle> load_obj(std::string const &filename)
{
	std::vector<triangle> ret;

	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;

	std::string err;

	if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &err, filename.c_str()))
	{
		throw std::runtime_error("failed to load obj \"" + filename + "\"");
	}

	int i = 0;
	triangle t;

	for (auto const &shape : shapes)
	{
		for (auto const &index : shape.mesh.indices)
		{
			vertex v;

			v.pos = vec3(attrib.vertices[3 * index.vertex_index], attrib.vertices[3 * index.vertex_index + 1], attrib.vertices[3 * index.vertex_index + 2]);
		
			if (attrib.normals.size())
				v.nrm = vec3(attrib.normals[3 * index.normal_index], attrib.normals[3 * index.normal_index + 1], attrib.normals[3 * index.normal_index + 2]);

			if (attrib.texcoords.size())
				v.uv = vec2(attrib.texcoords[2 * index.texcoord_index], attrib.texcoords[2 * index.texcoord_index]);

			switch (i)
			{
			case 0: t.v0 = v; break;
			case 1: t.v1 = v; break;
			case 2: t.v2 = v; break;
			}

			i++;
			i %= 3;
		
			if (i == 0) ret.push_back(t);
		}
	}

	return ret;
}

int main()
{
	SDL_Event event;
	SDL_Window *sdl_window;
	SDL_Renderer *sdl_renderer;

	SDL_Init(SDL_INIT_VIDEO);
	SDL_CreateWindowAndRenderer(width, height, 0, &sdl_window, &sdl_renderer);
	SDL_Texture *sdl_framebuffer = SDL_CreateTexture(sdl_renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, width, height);
	SDL_SetWindowTitle(sdl_window, "rasterizer");

	std::printf("Model name: \n");
	std::string str;
	std::getline(std::cin, str);

	scene.push_back(model(load_obj(str), mat4(1)));

	scene.push_back(model(load_obj("./cursor.obj"), glm::translate(mat4(1), vec3(1, 2, 0))));
	mat4 &cursor_transform = scene[1].transform;

	auto begin = std::chrono::high_resolution_clock::now();

	bool exit = false;

	std::printf("%u objects.\n", scene.size());

	mat_projection = glm::perspective(glm::radians(fov), static_cast<float_type>(width) / static_cast<float_type>(height), clip_near, clip_far);

	std::chrono::milliseconds::rep time_last;

	bool key_w = false;
	bool key_s = false;
	bool key_a = false;
	bool key_d = false;
	bool key_shift = false;
	bool key_space = false;
	bool key_up = false;
	bool key_down = false;
	
	vec2 old_cursor;
	vec2 camera_yaw_pitch(0);
	vec3 camera_pos(-1, 0, 0);
	vec3 camera_look_at(0, 0, 0);

	float_type view_radius = 5.;
	
	float_type const zoom_speed = 3.;
	float_type const move_speed = 8.;

	while (!exit)
	{
		// calculate time delta
		auto time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - begin).count();
		float_type t = time * .001;
		float_type delta = static_cast<float_type>(time - time_last) * .001;
		time_last = time;

		while (SDL_PollEvent(&event))
		{
			switch (event.type)
			{
			case SDL_QUIT:
				exit = true;
				continue;
				break;

			case SDL_KEYDOWN:
				switch (event.key.keysym.sym)
				{
				case SDLK_w:      key_w     = true; break;
				case SDLK_s:      key_s     = true; break;
				case SDLK_a:      key_a     = true; break;
				case SDLK_d:      key_d     = true; break;
				case SDLK_LSHIFT: key_shift = true; break;
				case SDLK_SPACE:  key_space = true; break;
				case SDLK_UP:     key_up    = true; break;
				case SDLK_DOWN:   key_down  = true; break;
				}
				break;

			case SDL_KEYUP:
				switch (event.key.keysym.sym)
				{
				case SDLK_w:      key_w     = false; break;
				case SDLK_s:      key_s     = false; break;
				case SDLK_a:      key_a     = false; break;
				case SDLK_d:      key_d     = false; break;
				case SDLK_LSHIFT: key_shift = false; break;
				case SDLK_SPACE:  key_space = false; break;
				case SDLK_UP:     key_up    = false; break;
				case SDLK_DOWN:   key_down  = false; break;
				}
				break;

			case SDL_MOUSEWHEEL:
				if (event.wheel.y > 0 && (view_radius - delta * zoom_speed * 3) > .05) view_radius -= delta * zoom_speed * 3;
				if (event.wheel.y < 0) view_radius += delta * zoom_speed * 3;
				break;
			} 
		}

		viewport = vec4(clip_near, clip_far, width, height);

		// bind the framebuffer and depth buffer
		bind_buffers(main_image, depth_buffer);

		// clear the image
		clear_buffers();

		// update 
		int mx, my;
		SDL_GetMouseState(&mx, &my);
		vec2 cursor_pos(mx, my);
		camera_yaw_pitch += (cursor_pos - old_cursor) * .03f;
		old_cursor = cursor_pos;

		if (camera_yaw_pitch.x >  _2PI) camera_yaw_pitch.x -=  _2PI;
		if (camera_yaw_pitch.x < -_2PI) camera_yaw_pitch.x -= -_2PI;
		if (camera_yaw_pitch.y >  (M_PI_2 - .1)) camera_yaw_pitch.y =  (M_PI_2 - .1);
		if (camera_yaw_pitch.y < -(M_PI_2 - .1)) camera_yaw_pitch.y = -(M_PI_2 - .1);		

		if (key_up && (view_radius - delta * zoom_speed) > .05) view_radius -= delta * zoom_speed;
		if (key_down) view_radius += delta * zoom_speed;

		vec3 camera_direction = glm::normalize(camera_look_at - camera_pos);

		if (key_space) camera_look_at.y += delta * move_speed;
		if (key_shift) camera_look_at.y -= delta * move_speed;
		if (key_w) camera_look_at += camera_direction * delta * move_speed;
		if (key_s) camera_look_at -= camera_direction * delta * move_speed;
		if (key_a) camera_look_at -= glm::cross(camera_direction, glm::vec3(0, 1, 0)) * delta * move_speed;
		if (key_d) camera_look_at += glm::cross(camera_direction, glm::vec3(0, 1, 0)) * delta * move_speed;

		cursor_transform = glm::translate(mat4(1), camera_look_at);

		camera_pos.x = std::cos(camera_yaw_pitch.x) * std::cos(camera_yaw_pitch.y);
		camera_pos.y = std::sin(camera_yaw_pitch.y);
		camera_pos.z = std::sin(camera_yaw_pitch.x) * std::cos(camera_yaw_pitch.y);

		camera_pos *= view_radius;
		camera_pos += camera_look_at;
		
		// update view matrix
		mat_view = glm::lookAt(camera_pos, camera_look_at, vec3(0.f, 1.f, 0.f));

		// render
		for (auto &model : scene)
		{
			#pragma omp parallel for
			for (auto &tri : model.data) render_tri(tri, model);
		}
		
		SDL_UpdateTexture(sdl_framebuffer, 0, main_image.data(), width * 4);
		SDL_SetRenderDrawColor(sdl_renderer, 0, 0, 0, 0);
		SDL_RenderClear(sdl_renderer);
		SDL_RenderCopy(sdl_renderer, sdl_framebuffer, 0, 0);
		SDL_RenderPresent(sdl_renderer);

		//std::printf("%.2ffps (%.4fs)\n", 1. / delta, delta);
	}

	SDL_Surface *surface = SDL_CreateRGBSurfaceWithFormat(0, width, height, 32, SDL_PIXELFORMAT_ARGB8888);
	SDL_RenderReadPixels(sdl_renderer, 0, SDL_PIXELFORMAT_ARGB8888, surface->pixels, surface->pitch);
	auto t = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	SDL_SaveBMP(surface, ("screenshots/" + std::to_string(t) + ".bmp").c_str());

	SDL_DestroyTexture(sdl_framebuffer);
	SDL_DestroyRenderer(sdl_renderer);
	SDL_DestroyWindow(sdl_window);
	SDL_Quit();

	return 0;
}
